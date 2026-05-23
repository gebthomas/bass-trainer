"""Tests for core/session_log.py — session telemetry model.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Construction
  1.  Valid empty log (no events, no metrics, no optional paths).
  2.  Valid log with events.
  3.  Defaults: all optional fields are None or empty.

Validation — schema_version
  4.  schema_version != 1 raises ValueError.

Validation — started_at
  5.  Empty started_at raises ValueError.
  6.  Whitespace-only started_at raises ValueError.

Validation — events: time_sec
  7.  Event with time_sec < 0 raises ValueError.
  8.  Event with time_sec == 0 is valid (boundary).

Validation — events: event_type
  9.  Event with empty event_type raises ValueError.
  10. Event with whitespace-only event_type raises ValueError.
  10b. Valid target_hit accepted.
  10c. Valid target_miss accepted.
  10d. Valid extra_onset accepted.
  10e. Unknown event_type raises ValueError.

Validation — events: target_index
  11. Event with target_index < 0 raises ValueError.
  12. Event with target_index == 0 is valid (boundary).

Validation — events: metadata
  13. Event metadata with non-string value raises ValueError.

Validation — metrics
  14. Metric with string value raises ValueError.
  15. Metric with int value is valid.
  16. Metric with float value is valid.

Validation — log metadata
  17. Log metadata with non-string value raises ValueError.

Serialisation
  18. session_log_from_dict raises KeyError on missing schema_version.
  19. session_log_from_dict raises KeyError on missing started_at.
  20. dict round-trip equality (empty log).
  21. JSON round-trip equality (empty log).
  22. JSON round-trip equality (log with events and metrics).
  23. Optional path fields preserved through JSON round-trip.

File I/O
  24. save_session_log_file + load_session_log_file round-trip equality.

Metadata
  25. Log metadata preserved through JSON round-trip.
  26. Event metadata preserved through JSON round-trip.

Metrics
  27. Metrics preserved through JSON round-trip.
  28. Int metric value survives round-trip as a number.

append_event
  29. Valid event is appended to the log.
  30. Return value is the log itself.
  31. Invalid event (negative time) raises ValueError and is not appended.
  32. Invalid event (empty event_type) raises ValueError and is not appended.
  33. Multiple appends accumulate events in order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_log import (
    ALLOWED_EVENT_TYPES,
    EXTRA_ONSET,
    SCHEMA_VERSION,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
    append_event,
    load_session_log_file,
    save_session_log_file,
    session_log_from_dict,
    session_log_from_json,
    session_log_to_dict,
    session_log_to_json,
    validate_session_log,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _minimal_dict() -> dict:
    return {"schema_version": 1, "started_at": "2026-05-23T10:00:00"}


def _minimal_event() -> SessionEvent:
    return SessionEvent(time_sec=0.5, event_type=TARGET_HIT, target_index=0)


def _minimal_log() -> SessionLog:
    return session_log_from_dict(_minimal_dict())


# ── 1–3: Construction ─────────────────────────────────────────────────────────

def test_valid_empty_log():
    log = session_log_from_dict(_minimal_dict())
    assert log.schema_version == 1
    assert log.started_at     == "2026-05-23T10:00:00"
    assert log.events         == []
    assert log.metrics        == {}
    assert log.metadata       == {}


def test_valid_log_with_events():
    data = _minimal_dict()
    data["events"] = [
        {"time_sec": 0.5, "event_type": "target_hit",  "target_index": 0},
        {"time_sec": 1.0, "event_type": "target_miss", "target_index": 1},
    ]
    log = session_log_from_dict(data)
    assert len(log.events)          == 2
    assert log.events[0].event_type == TARGET_HIT
    assert log.events[1].event_type == TARGET_MISS


def test_defaults():
    log = _minimal_log()
    assert log.ended_at           is None
    assert log.practice_mode_path is None
    assert log.exercise_path      is None
    assert log.alignment_path     is None
    assert log.events             == []
    assert log.metrics            == {}
    assert log.metadata           == {}


# ── 4: schema_version ────────────────────────────────────────────────────────

def test_schema_version_2_raises():
    data = _minimal_dict()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        session_log_from_dict(data)


# ── 5–6: started_at ──────────────────────────────────────────────────────────

def test_empty_started_at_raises():
    log = SessionLog(schema_version=1, started_at="")
    with pytest.raises(ValueError, match="started_at"):
        validate_session_log(log)


def test_whitespace_started_at_raises():
    log = SessionLog(schema_version=1, started_at="   ")
    with pytest.raises(ValueError, match="started_at"):
        validate_session_log(log)


# ── 7–8: event time_sec ───────────────────────────────────────────────────────

def test_event_negative_time_raises():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=-0.1, event_type=TARGET_HIT))
    with pytest.raises(ValueError, match="time_sec"):
        validate_session_log(log)


def test_event_zero_time_valid():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.0, event_type=TARGET_HIT))
    validate_session_log(log)  # must not raise


# ── 9–10: event event_type ────────────────────────────────────────────────────

def test_event_empty_type_raises():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type=""))
    with pytest.raises(ValueError, match="event_type"):
        validate_session_log(log)


def test_event_whitespace_type_raises():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type="  "))
    with pytest.raises(ValueError, match="event_type"):
        validate_session_log(log)


# ── 11–12: event target_index ────────────────────────────────────────────────

def test_event_negative_target_index_raises():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type=TARGET_HIT, target_index=-1))
    with pytest.raises(ValueError, match="target_index"):
        validate_session_log(log)


def test_event_zero_target_index_valid():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type=TARGET_HIT, target_index=0))
    validate_session_log(log)  # must not raise


# ── 13: event metadata ───────────────────────────────────────────────────────

def test_event_metadata_non_string_raises():
    event = SessionEvent(
        time_sec=0.5,
        event_type=TARGET_HIT,
        metadata={"score": 42},  # type: ignore[dict-item]
    )
    log = _minimal_log()
    log.events.append(event)
    with pytest.raises(ValueError, match="metadata"):
        validate_session_log(log)


# ── 14–16: metrics ───────────────────────────────────────────────────────────

def test_metric_string_value_raises():
    log = _minimal_log()
    log.metrics["hit_rate"] = "high"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="metrics"):
        validate_session_log(log)


def test_metric_int_value_valid():
    log = _minimal_log()
    log.metrics["n_hits"] = 4
    validate_session_log(log)  # must not raise


def test_metric_float_value_valid():
    log = _minimal_log()
    log.metrics["hit_rate"] = 0.75
    validate_session_log(log)  # must not raise


# ── 17: log metadata ─────────────────────────────────────────────────────────

def test_log_metadata_non_string_raises():
    log = SessionLog(
        schema_version=1,
        started_at="2026-05-23T10:00:00",
        metadata={"bpm": 120},  # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_session_log(log)


# ── 18–19: missing required fields ───────────────────────────────────────────

def test_from_dict_missing_schema_version_raises():
    data = _minimal_dict()
    del data["schema_version"]
    with pytest.raises(KeyError):
        session_log_from_dict(data)


def test_from_dict_missing_started_at_raises():
    data = _minimal_dict()
    del data["started_at"]
    with pytest.raises(KeyError):
        session_log_from_dict(data)


# ── 20–23: serialisation ─────────────────────────────────────────────────────

def test_dict_roundtrip_empty_log():
    log  = _minimal_log()
    log2 = session_log_from_dict(session_log_to_dict(log))
    assert log == log2


def test_json_roundtrip_empty_log():
    log  = _minimal_log()
    log2 = session_log_from_json(session_log_to_json(log))
    assert log == log2


def test_json_roundtrip_log_with_events_and_metrics():
    data = _minimal_dict()
    data["events"] = [
        {"time_sec": 0.5, "event_type": "target_hit",  "target_index": 0, "value": -0.02},
        {"time_sec": 1.0, "event_type": "target_miss", "target_index": 1},
    ]
    data["metrics"] = {"hit_rate": 0.5, "mean_timing_error_ms": 20.0}
    log  = session_log_from_dict(data)
    log2 = session_log_from_json(session_log_to_json(log))
    assert log == log2
    assert log2.events[0].value       == pytest.approx(-0.02)
    assert log2.metrics["hit_rate"]   == pytest.approx(0.5)


def test_optional_paths_preserved_through_json():
    data = _minimal_dict()
    data["ended_at"]           = "2026-05-23T10:30:00"
    data["practice_mode_path"] = "practice_modes/walking_bass.json"
    data["exercise_path"]      = "exercises/walking_bass_Am.json"
    data["alignment_path"]     = "alignments/autumn_leaves.json"
    log  = session_log_from_dict(data)
    log2 = session_log_from_json(session_log_to_json(log))
    assert log2.ended_at           == "2026-05-23T10:30:00"
    assert log2.practice_mode_path == "practice_modes/walking_bass.json"
    assert log2.exercise_path      == "exercises/walking_bass_Am.json"
    assert log2.alignment_path     == "alignments/autumn_leaves.json"


# ── 24: file I/O ──────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    data = _minimal_dict()
    data["events"]  = [{"time_sec": 0.5, "event_type": "target_hit", "target_index": 0}]
    data["metrics"] = {"hit_rate": 1.0}
    log  = session_log_from_dict(data)
    dest = tmp_path / "sessions" / "test.json"
    save_session_log_file(log, dest)
    log2 = load_session_log_file(dest)
    assert log == log2


# ── 25–26: metadata preservation ─────────────────────────────────────────────

def test_log_metadata_preserved():
    data = _minimal_dict()
    data["metadata"] = {"instrument": "upright bass", "venue": "home studio"}
    log  = session_log_from_dict(data)
    log2 = session_log_from_json(session_log_to_json(log))
    assert log2.metadata == {"instrument": "upright bass", "venue": "home studio"}


def test_event_metadata_preserved():
    data = _minimal_dict()
    data["events"] = [
        {
            "time_sec": 0.5,
            "event_type": "target_hit",
            "metadata": {"detected_pitch": "E1", "string": "E"},
        }
    ]
    log  = session_log_from_dict(data)
    log2 = session_log_from_json(session_log_to_json(log))
    assert log2.events[0].metadata == {"detected_pitch": "E1", "string": "E"}


# ── 27–28: metrics preservation ──────────────────────────────────────────────

def test_metrics_preserved_through_json():
    data = _minimal_dict()
    data["metrics"] = {"hit_rate": 0.75, "mean_abs_timing_ms": 18.5}
    log  = session_log_from_dict(data)
    log2 = session_log_from_json(session_log_to_json(log))
    assert log2.metrics["hit_rate"]            == pytest.approx(0.75)
    assert log2.metrics["mean_abs_timing_ms"]  == pytest.approx(18.5)


def test_int_metric_survives_roundtrip():
    data = _minimal_dict()
    data["metrics"] = {"n_hits": 4}
    log  = session_log_from_dict(data)
    log2 = session_log_from_json(session_log_to_json(log))
    assert log2.metrics["n_hits"] == 4


# ── 29–33: append_event ───────────────────────────────────────────────────────

def test_append_event_adds_event():
    log   = _minimal_log()
    event = _minimal_event()
    append_event(log, event)
    assert len(log.events) == 1
    assert log.events[0]   is event


def test_append_event_returns_log():
    log    = _minimal_log()
    result = append_event(log, _minimal_event())
    assert result is log


def test_append_event_invalid_time_raises_and_does_not_append():
    log   = _minimal_log()
    bad   = SessionEvent(time_sec=-1.0, event_type=TARGET_HIT)
    with pytest.raises(ValueError, match="time_sec"):
        append_event(log, bad)
    assert log.events == []


def test_append_event_invalid_type_raises_and_does_not_append():
    log = _minimal_log()
    bad = SessionEvent(time_sec=0.5, event_type="")
    with pytest.raises(ValueError, match="event_type"):
        append_event(log, bad)
    assert log.events == []


def test_append_event_accumulates_in_order():
    log = _minimal_log()
    ev1 = SessionEvent(time_sec=0.5, event_type=TARGET_HIT,  target_index=0)
    ev2 = SessionEvent(time_sec=1.0, event_type=TARGET_MISS, target_index=1)
    ev3 = SessionEvent(time_sec=1.5, event_type=TARGET_HIT,  target_index=2)
    append_event(log, ev1)
    append_event(log, ev2)
    append_event(log, ev3)
    assert len(log.events) == 3
    assert [e.event_type for e in log.events] == [TARGET_HIT, TARGET_MISS, TARGET_HIT]
    assert [e.target_index for e in log.events] == [0, 1, 2]


# ── 10b–10e: event_type vocabulary ───────────────────────────────────────────

def test_valid_target_hit_accepted():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type=TARGET_HIT))
    validate_session_log(log)  # must not raise


def test_valid_target_miss_accepted():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type=TARGET_MISS))
    validate_session_log(log)  # must not raise


def test_valid_extra_onset_accepted():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type=EXTRA_ONSET))
    validate_session_log(log)  # must not raise


def test_unknown_event_type_raises():
    log = _minimal_log()
    log.events.append(SessionEvent(time_sec=0.5, event_type="hit"))
    with pytest.raises(ValueError, match="event_type"):
        validate_session_log(log)
