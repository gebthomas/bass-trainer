"""Tests for extract_timing_errors() in scripts/practice_session_demo.py.

All tests are pure — no audio hardware, no matplotlib display.

Test matrix
-----------
extract_timing_errors
    1.  Empty log (no events) → [].
    2.  Single TARGET_HIT → (target_index, timing_error_ms).
    3.  Single TARGET_MISS → (target_index, None).
    4.  EXTRA_ONSET events are excluded from the result.
    5.  Mixed hits and misses → correct pairs for each.
    6.  Result is sorted by target_index regardless of log order.
    7.  Seconds are converted to milliseconds (× 1000).
    8.  Negative timing error (early onset) preserved with sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT    = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from practice_session_demo import extract_timing_errors
from core.session_log import (
    EXTRA_ONSET,
    TARGET_HIT,
    TARGET_MISS,
    SessionEvent,
    SessionLog,
)


def _make_log(*events: SessionEvent) -> SessionLog:
    log = SessionLog(schema_version="1", started_at="2026-01-01T00:00:00+00:00")
    log.events.extend(events)
    return log


def _hit(target_index: int, value_s: float, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=TARGET_HIT,
        target_index=target_index,
        value=value_s,
    )


def _miss(target_index: int, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=TARGET_MISS,
        target_index=target_index,
        value=None,
    )


def _extra(time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=EXTRA_ONSET,
        target_index=None,
        value=None,
    )


# ── extract_timing_errors ─────────────────────────────────────────────────────

def test_empty_log_returns_empty():
    assert extract_timing_errors(_make_log()) == []


def test_single_hit_returns_index_and_ms():
    log = _make_log(_hit(0, value_s=0.050))
    result = extract_timing_errors(log)
    assert result == [(0, pytest.approx(50.0))]


def test_single_miss_returns_index_and_none():
    log = _make_log(_miss(2))
    result = extract_timing_errors(log)
    assert result == [(2, None)]


def test_extra_onset_excluded():
    log = _make_log(_extra(), _hit(0, value_s=0.030))
    result = extract_timing_errors(log)
    assert len(result) == 1
    assert result[0][0] == 0


def test_mixed_hits_and_misses():
    log = _make_log(_hit(0, 0.020), _miss(1), _hit(2, -0.010))
    result = extract_timing_errors(log)
    assert result[0] == (0, pytest.approx(20.0))
    assert result[1] == (1, None)
    assert result[2] == (2, pytest.approx(-10.0))


def test_sorted_by_target_index():
    # Log order: 2, 0, 1 — result must be 0, 1, 2
    log = _make_log(_hit(2, 0.010), _miss(0), _hit(1, 0.005))
    result = extract_timing_errors(log)
    indices = [idx for idx, _ in result]
    assert indices == [0, 1, 2]


def test_seconds_converted_to_milliseconds():
    log = _make_log(_hit(0, value_s=0.123))
    _, err_ms = extract_timing_errors(log)[0]
    assert err_ms == pytest.approx(123.0)


def test_negative_timing_error_preserved():
    log = _make_log(_hit(0, value_s=-0.045))
    _, err_ms = extract_timing_errors(log)[0]
    assert err_ms == pytest.approx(-45.0)
