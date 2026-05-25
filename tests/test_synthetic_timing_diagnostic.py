"""Tests for scripts/synthetic_timing_diagnostic.py — run_offset_check helper.

run_offset_check is pure Python (no sounddevice, no aubio) and delegates to
session_replay.replay_session_data(), so it can be tested without mocking.

Test matrix
-----------
Accuracy
    1.  Zero offset → reported error is zero for all targets.
    2.  +30 ms offset → reported error is +30 ms (within floating-point noise).
    3.  -80 ms offset → reported error is -80 ms.
    4.  +150 ms offset → within match window at 60 BPM; hits reported correctly.

Match-window behaviour
    5.  Onset beyond match window → severity is "MISSED", reported_ms is None.
    6.  Onset just inside match window → hit, not missed.

Row structure
    7.  Result count equals target count.
    8.  target_time_s is the correct nominal beat time (count_in + beat * beat_s).
    9.  onset_time_s equals target_time_s + offset_s.
    10. delta_ms equals reported_ms − injected_ms.

Severity labels
    11. 0 ms offset → severity "good".
    12. 80 ms offset → severity "warn".
    13. 150 ms offset → severity "miss" (but still a hit — within match window).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.synthetic_timing_diagnostic as _diag

run_offset_check = _diag.run_offset_check

_BPM     = 60.0
_COUNT   = 2
_TARGETS = [{"time": i} for i in range(4)]
_SINGLE  = [{"time": 0}]

_FLOAT_TOL = 1e-6   # ms; timing error should be exact to float precision


class TestAccuracy:
    def test_zero_offset_reports_zero_for_all_targets(self):
        rows = run_offset_check(_BPM, _COUNT, _TARGETS, 0.0)
        for row in rows:
            assert row["severity"] != "MISSED"
            assert abs(row["reported_ms"]) < _FLOAT_TOL
            assert abs(row["delta_ms"])    < _FLOAT_TOL

    def test_plus_30ms_offset_reported_correctly(self):
        rows = run_offset_check(_BPM, _COUNT, _TARGETS, 30.0)
        for row in rows:
            assert row["severity"] != "MISSED"
            assert abs(row["reported_ms"] - 30.0) < _FLOAT_TOL
            assert abs(row["delta_ms"])            < _FLOAT_TOL

    def test_minus_80ms_offset_reported_correctly(self):
        rows = run_offset_check(_BPM, _COUNT, _TARGETS, -80.0)
        for row in rows:
            assert row["severity"] != "MISSED"
            assert abs(row["reported_ms"] - (-80.0)) < _FLOAT_TOL
            assert abs(row["delta_ms"])               < _FLOAT_TOL

    def test_plus_150ms_within_window_at_60bpm(self):
        # At 60 BPM: match_window = clamp(30/60, 0.10, 0.35) = 0.35 s = 350 ms
        # +150 ms is within window; all targets should be hits
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, 150.0)
        assert rows[0]["severity"] != "MISSED"
        assert abs(rows[0]["reported_ms"] - 150.0) < _FLOAT_TOL

    def test_minus_150ms_within_window_at_60bpm(self):
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, -150.0)
        assert rows[0]["severity"] != "MISSED"
        assert abs(rows[0]["reported_ms"] - (-150.0)) < _FLOAT_TOL


class TestMatchWindowBehaviour:
    def test_onset_beyond_window_is_missed(self):
        # At 60 BPM match_window = 350 ms; 400 ms exceeds it
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, 400.0)
        assert rows[0]["severity"]    == "MISSED"
        assert rows[0]["reported_ms"] is None
        assert rows[0]["delta_ms"]    is None

    def test_onset_just_inside_window_is_hit(self):
        # 340 ms < 350 ms match window at 60 BPM
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, 340.0)
        assert rows[0]["severity"] != "MISSED"
        assert rows[0]["reported_ms"] is not None


class TestRowStructure:
    def test_result_count_equals_target_count(self):
        targets = [{"time": i} for i in range(6)]
        rows    = run_offset_check(_BPM, _COUNT, targets, 0.0)
        assert len(rows) == 6

    def test_target_time_s_is_nominal_beat_time(self):
        # count_in=2, bpm=60 → beat_s=1.0, count_in_s=2.0
        # target 0 → 2.0 s, target 1 → 3.0 s, target 2 → 4.0 s
        rows = run_offset_check(60.0, 2, [{"time": 0}, {"time": 1}, {"time": 2}], 0.0)
        assert abs(rows[0]["target_time_s"] - 2.0) < 1e-9
        assert abs(rows[1]["target_time_s"] - 3.0) < 1e-9
        assert abs(rows[2]["target_time_s"] - 4.0) < 1e-9

    def test_onset_time_s_equals_target_plus_offset(self):
        rows = run_offset_check(60.0, 2, _SINGLE, 50.0)
        expected = 2.0 + 0.050   # target at 2.0 s + 50 ms
        assert abs(rows[0]["onset_time_s"] - expected) < 1e-9

    def test_delta_ms_equals_reported_minus_injected(self):
        rows = run_offset_check(_BPM, _COUNT, _TARGETS, 30.0)
        for row in rows:
            if row["delta_ms"] is not None:
                assert abs(row["delta_ms"] - (row["reported_ms"] - row["injected_ms"])) < 1e-9

    def test_target_idx_matches_position(self):
        rows = run_offset_check(_BPM, _COUNT, _TARGETS, 0.0)
        for i, row in enumerate(rows):
            assert row["target_idx"] == i


class TestSeverityLabels:
    def test_zero_offset_is_good(self):
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, 0.0)
        assert rows[0]["severity"] == "good"

    def test_80ms_offset_is_warn(self):
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, 80.0)
        assert rows[0]["severity"] == "warn"

    def test_negative_80ms_is_also_warn(self):
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, -80.0)
        assert rows[0]["severity"] == "warn"

    def test_150ms_offset_is_miss_severity_but_still_hit(self):
        # 150 ms > 120 ms threshold → severity "miss", but the onset was matched
        rows = run_offset_check(_BPM, _COUNT, _SINGLE, 150.0)
        assert rows[0]["severity"]    == "miss"
        assert rows[0]["reported_ms"] is not None
