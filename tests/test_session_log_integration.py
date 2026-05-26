"""Tests for the session-log conversion helpers in live_feedback_demo.py.

sounddevice is mocked so no audio hardware is required.
Tests only cover the pure functions event_from_pipeline_result and
session_events_from_run; no file I/O is exercised.

Test matrix
-----------
event_from_pipeline_result — TARGET_HIT
    1.  event_type is TARGET_HIT when timing_error_s is not None.
    2.  target_index is preserved.
    3.  value equals timing_error_s.
    4.  time_sec = target_beat_s + timing_error_s (late onset).
    5.  time_sec = target_beat_s + timing_error_s (early onset).
    6.  time_sec clamped to 0.0 when computed value would be negative.

event_from_pipeline_result — TARGET_MISS
    7.  event_type is TARGET_MISS when timing_error_s is None.
    8.  target_index is preserved.
    9.  value is None.
    10. time_sec equals the nominal beat time (not the onset time).

event_from_pipeline_result — time_sec formula
    11. beat_s = 60/bpm is used correctly (bpm=120 → beat_s=0.5).
    12. count_in beats shift time_sec by count_in * beat_s.
    13. target["time"] beats shift time_sec by target_time * beat_s.

session_events_from_run — event assembly
    14. Returns one event per pipeline event when unevaluated is empty.
    15. Unevaluated indices add TARGET_MISS events for the right targets.
    16. Unevaluated events have time_sec equal to nominal beat time.
    17. Events are sorted ascending by time_sec.
    18. Empty pipeline_events + empty unevaluated → empty list.
    19. Mixed evaluated (hit + miss) + unevaluated → correct counts.

Argument parsing
    20. Default save_session_log is None.
    21. --save-session-log DIR sets save_session_log to "DIR".
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Import demo module without triggering sounddevice hardware queries ─────────

_sd_mock = unittest.mock.MagicMock()
_sd_mock.query_devices.return_value = []

with unittest.mock.patch.dict("sys.modules", {"sounddevice": _sd_mock}):
    import importlib
    import scripts.live_feedback_demo as _demo_mod
    importlib.reload(_demo_mod)

event_from_pipeline_result = _demo_mod.event_from_pipeline_result
session_events_from_run    = _demo_mod.session_events_from_run
_parse_args                = _demo_mod._parse_args

from core.session_log import TARGET_HIT, TARGET_MISS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(*argv: str):
    with unittest.mock.patch("sys.argv", ["live_feedback_demo.py", *argv]):
        return _parse_args()


def _hit_event(idx: int, timing_error_s: float) -> dict:
    return {"target_index": idx, "timing_error_s": timing_error_s}


def _miss_event(idx: int) -> dict:
    return {"target_index": idx, "timing_error_s": None}


def _targets(n: int) -> list[dict]:
    return [{"time": float(i)} for i in range(n)]


def _beat_time(idx: int, bpm: float, count_in: int) -> float:
    return count_in * (60.0 / bpm) + idx * (60.0 / bpm)


# ── event_from_pipeline_result — TARGET_HIT ──────────────────────────────────

class TestEventFromPipelineResultHit:

    BPM      = 60.0
    COUNT_IN = 2
    TARGETS  = _targets(4)
    BEAT_S   = 60.0 / BPM

    def test_event_type_is_target_hit(self):
        ev = event_from_pipeline_result(_hit_event(0, 0.020), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.event_type == TARGET_HIT

    def test_target_index_preserved(self):
        ev = event_from_pipeline_result(_hit_event(2, 0.010), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.target_index == 2

    def test_value_equals_timing_error(self):
        ev = event_from_pipeline_result(_hit_event(0, -0.025), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.value == pytest.approx(-0.025)

    def test_time_sec_late_onset(self):
        error_s      = +0.030
        target_beat_s = _beat_time(0, self.BPM, self.COUNT_IN)
        ev = event_from_pipeline_result(_hit_event(0, error_s), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.time_sec == pytest.approx(target_beat_s + error_s)

    def test_time_sec_early_onset(self):
        error_s      = -0.020
        target_beat_s = _beat_time(1, self.BPM, self.COUNT_IN)
        ev = event_from_pipeline_result(_hit_event(1, error_s), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.time_sec == pytest.approx(target_beat_s + error_s)

    def test_time_sec_clamped_to_zero_when_negative(self):
        # count_in=0, target 0 at 0.0s, error=-0.5 → would be -0.5 → clamped to 0.
        targets = _targets(2)
        ev = event_from_pipeline_result(_hit_event(0, -0.5), targets, 60.0, count_in=0)
        assert ev.time_sec == pytest.approx(0.0)


# ── event_from_pipeline_result — TARGET_MISS ─────────────────────────────────

class TestEventFromPipelineResultMiss:

    BPM      = 60.0
    COUNT_IN = 2
    TARGETS  = _targets(4)

    def test_event_type_is_target_miss(self):
        ev = event_from_pipeline_result(_miss_event(0), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.event_type == TARGET_MISS

    def test_target_index_preserved(self):
        ev = event_from_pipeline_result(_miss_event(3), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.target_index == 3

    def test_value_is_none(self):
        ev = event_from_pipeline_result(_miss_event(0), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.value is None

    def test_time_sec_equals_nominal_beat_time(self):
        target_beat_s = _beat_time(1, self.BPM, self.COUNT_IN)
        ev = event_from_pipeline_result(_miss_event(1), self.TARGETS, self.BPM, self.COUNT_IN)
        assert ev.time_sec == pytest.approx(target_beat_s)


# ── event_from_pipeline_result — time_sec formula ────────────────────────────

class TestTimingFormula:

    def test_bpm_120_beat_s_is_half_second(self):
        targets = _targets(2)
        ev      = event_from_pipeline_result(_hit_event(0, 0.0), targets, 120.0, count_in=1)
        assert ev.time_sec == pytest.approx(0.5)  # 1 beat × 0.5 s/beat

    def test_count_in_shifts_time_sec(self):
        targets = _targets(2)
        ev2     = event_from_pipeline_result(_hit_event(0, 0.0), targets, 60.0, count_in=2)
        ev4     = event_from_pipeline_result(_hit_event(0, 0.0), targets, 60.0, count_in=4)
        assert ev4.time_sec - ev2.time_sec == pytest.approx(2.0)  # 2 extra beats

    def test_target_time_shifts_time_sec(self):
        targets = [{"time": 0.0}, {"time": 3.0}]
        ev0 = event_from_pipeline_result(_hit_event(0, 0.0), targets, 60.0, count_in=0)
        ev1 = event_from_pipeline_result(_hit_event(1, 0.0), targets, 60.0, count_in=0)
        assert ev1.time_sec - ev0.time_sec == pytest.approx(3.0)


# ── session_events_from_run ───────────────────────────────────────────────────

class TestSessionEventsFromRun:

    BPM      = 60.0
    COUNT_IN = 2
    TARGETS  = _targets(4)

    def _run(self, pipeline_events, unevaluated):
        return session_events_from_run(
            pipeline_events, unevaluated, self.TARGETS, self.BPM, self.COUNT_IN,
        )

    def test_one_event_per_pipeline_event_when_no_unevaluated(self):
        pipeline = [_hit_event(0, 0.01), _miss_event(1)]
        evts     = self._run(pipeline, set())
        assert len(evts) == 2

    def test_unevaluated_adds_target_miss_events(self):
        pipeline = [_hit_event(0, 0.01)]
        evts     = self._run(pipeline, {2, 3})
        miss_types = [e.event_type for e in evts if e.target_index in {2, 3}]
        assert all(t == TARGET_MISS for t in miss_types)
        assert len(miss_types) == 2

    def test_unevaluated_event_time_sec_is_nominal_beat(self):
        evts = self._run([], {1})
        ev   = next(e for e in evts if e.target_index == 1)
        assert ev.time_sec == pytest.approx(_beat_time(1, self.BPM, self.COUNT_IN))

    def test_events_sorted_ascending_by_time_sec(self):
        # Beat 3 hit (late, +0.05), beat 0 miss — beat 0 should come first.
        pipeline = [_hit_event(3, 0.05), _miss_event(0)]
        evts     = self._run(pipeline, set())
        times    = [e.time_sec for e in evts]
        assert times == sorted(times)

    def test_empty_inputs_empty_result(self):
        assert self._run([], set()) == []

    def test_total_count_correct(self):
        # 2 pipeline events + 2 unevaluated → 4 total
        pipeline = [_hit_event(0, 0.01), _miss_event(1)]
        evts     = self._run(pipeline, {2, 3})
        assert len(evts) == 4

    def test_all_hits_in_pipeline(self):
        pipeline = [_hit_event(i, float(i) * 0.01) for i in range(4)]
        evts     = self._run(pipeline, set())
        assert all(e.event_type == TARGET_HIT for e in evts)

    def test_mixed_hit_and_miss_correct_types(self):
        pipeline = [_hit_event(0, 0.01), _miss_event(1), _hit_event(2, -0.02)]
        evts     = self._run(pipeline, {3})
        types    = {e.target_index: e.event_type for e in evts}
        assert types[0] == TARGET_HIT
        assert types[1] == TARGET_MISS
        assert types[2] == TARGET_HIT
        assert types[3] == TARGET_MISS


# ── Argument parsing ──────────────────────────────────────────────────────────

def test_default_save_session_log_is_none():
    assert _parse().save_session_log is None


def test_save_session_log_flag_sets_dir():
    args = _parse("--save-session-log", "sessions/test")
    assert args.save_session_log == "sessions/test"
