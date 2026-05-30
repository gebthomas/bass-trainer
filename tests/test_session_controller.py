"""Tests for core/session_controller.py

All tests are synthetic — no audio hardware required.

Test matrix
-----------
Phase transitions
    1.  Initial phase is WAITING.
    2.  start() transitions WAITING → COUNT_IN.
    3.  start() called twice raises RuntimeError.
    4.  update() in WAITING returns empty list.
    5.  update() before count-in ends stays in COUNT_IN, returns [].
    6.  update() at count-in boundary transitions to ACTIVE.
    7.  update() just past count-in boundary transitions to ACTIVE.

Completion via misses
    8.  All targets missed → phase becomes COMPLETE.
    9.  update() after COMPLETE returns [].
    10. is_complete() is False until all targets evaluated.
    11. is_complete() is True after COMPLETE.

Completion via hits
    12. All targets hit → phase becomes COMPLETE.
    13. Mixed hit + miss → COMPLETE when all targets evaluated.

Abort
    14. abort() from WAITING → ABORTED.
    15. abort() from COUNT_IN → ABORTED.
    16. abort() from ACTIVE → ABORTED.
    17. abort() from COMPLETE → no-op, phase stays COMPLETE.
    18. abort() from ABORTED → no-op.
    19. update() after ABORTED returns [].

Summary
    20. summary() returns None before session ends.
    21. summary() returns LogMetrics after COMPLETE.
    22. summary() returns LogMetrics after ABORTED (with events logged).
    23. summary() returns None when start() was never called.

Feedback event ordering and content
    24. Miss events from a single update() are in ascending target index order.
    25. Hit event has severity, messages, timing_error_s, target_idx fields.
    26. Exact hit has severity == "good".
    27. Miss event has severity == "miss".

Onset during count-in
    28. Onset passed during COUNT_IN transition is processed once ACTIVE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_controller import SessionController, SessionPhase
from core.log_metrics import LogMetrics

# ── Shared constants ──────────────────────────────────────────────────────────

BPM        = 120.0
COUNT_IN   = 2
SR         = 44100

BEAT_S     = 60.0 / BPM              # 0.5 s
COUNT_IN_S = COUNT_IN * BEAT_S       # 1.0 s
COUNT_IN_SAMPLE = round(COUNT_IN_S * SR)  # 44100

# Two targets: beat 0 (t=1.0s) and beat 1 (t=1.5s) relative to session start
TARGETS = [
    {"time": 0.0, "note": "E2", "label": "1"},
    {"time": 1.0, "note": "A2", "label": "2"},
]

# match_window_s(120) = 30/120 = 0.25s
MATCH_WINDOW_S = 0.25

# Nominal and deadline times (session-relative seconds)
NOM_0 = COUNT_IN_S + TARGETS[0]["time"] * BEAT_S  # 1.0s
NOM_1 = COUNT_IN_S + TARGETS[1]["time"] * BEAT_S  # 1.5s
DL_0  = NOM_0 + MATCH_WINDOW_S                    # 1.25s
DL_1  = NOM_1 + MATCH_WINDOW_S                    # 1.75s

PAST_ALL_S      = 2.0
PAST_ALL_SAMPLE = round(PAST_ALL_S * SR)           # 88200


def _make() -> SessionController:
    """Return a fresh controller in WAITING phase."""
    return SessionController(
        targets        = TARGETS,
        bpm            = BPM,
        count_in_beats = COUNT_IN,
        sample_rate    = SR,
    )


def _started() -> SessionController:
    c = _make()
    c.start()
    return c


# ── Phase transitions ─────────────────────────────────────────────────────────

def test_initial_phase_is_waiting():
    assert _make().phase == SessionPhase.WAITING


def test_start_transitions_to_count_in():
    c = _make()
    c.start()
    assert c.phase == SessionPhase.COUNT_IN


def test_start_twice_raises():
    c = _started()
    with pytest.raises(RuntimeError, match="only valid from WAITING"):
        c.start()


def test_update_in_waiting_returns_empty():
    c = _make()
    assert c.update(PAST_ALL_SAMPLE) == []
    assert c.phase == SessionPhase.WAITING


def test_update_before_count_in_boundary_stays_count_in():
    c = _started()
    result = c.update(COUNT_IN_SAMPLE - 1)
    assert c.phase == SessionPhase.COUNT_IN
    assert result == []


def test_update_at_count_in_boundary_transitions_to_active():
    c = _started()
    c.update(COUNT_IN_SAMPLE)
    assert c.phase == SessionPhase.ACTIVE


def test_update_past_count_in_boundary_transitions_to_active():
    c = _started()
    c.update(COUNT_IN_SAMPLE + 100)
    assert c.phase == SessionPhase.ACTIVE


# ── Completion via misses ─────────────────────────────────────────────────────

def test_all_targets_missed_produces_complete():
    c = _started()
    events = c.update(PAST_ALL_SAMPLE)
    assert c.phase == SessionPhase.COMPLETE
    assert len(events) == 2
    assert all(ev["severity"] == "miss" for ev in events)


def test_update_after_complete_returns_empty():
    c = _started()
    c.update(PAST_ALL_SAMPLE)
    assert c.is_complete()
    assert c.update(PAST_ALL_SAMPLE + SR) == []


def test_is_complete_false_mid_session():
    c = _started()
    # Past target 0 deadline but target 1 still pending
    c.update(round(DL_0 * SR) + 1)
    assert not c.is_complete()


def test_is_complete_true_after_all_evaluated():
    c = _started()
    c.update(PAST_ALL_SAMPLE)
    assert c.is_complete()


# ── Completion via hits ───────────────────────────────────────────────────────

def test_all_targets_hit_produces_complete():
    c = _started()
    # Hit target 0 at nominal time, then target 1
    c.update(round(1.1 * SR), [NOM_0])
    events = c.update(round(1.6 * SR), [NOM_1])
    assert c.is_complete()
    assert len(events) == 1
    assert events[0]["severity"] == "good"


def test_mixed_hit_and_miss_produces_complete():
    c = _started()
    # Hit target 0 at nominal, let target 1 be missed
    c.update(round(1.1 * SR), [NOM_0])
    events = c.update(PAST_ALL_SAMPLE)
    assert c.is_complete()
    # target 1 should be a miss event
    miss_events = [e for e in events if e["severity"] == "miss"]
    assert len(miss_events) == 1
    assert miss_events[0]["target_idx"] == 1


# ── Abort ─────────────────────────────────────────────────────────────────────

def test_abort_from_waiting():
    c = _make()
    c.abort()
    assert c.phase == SessionPhase.ABORTED


def test_abort_from_count_in():
    c = _started()
    assert c.phase == SessionPhase.COUNT_IN
    c.abort()
    assert c.phase == SessionPhase.ABORTED


def test_abort_from_active():
    c = _started()
    c.update(COUNT_IN_SAMPLE)
    assert c.phase == SessionPhase.ACTIVE
    c.abort()
    assert c.phase == SessionPhase.ABORTED


def test_abort_from_complete_is_noop():
    c = _started()
    c.update(PAST_ALL_SAMPLE)
    assert c.is_complete()
    c.abort()
    assert c.phase == SessionPhase.COMPLETE


def test_abort_from_aborted_is_noop():
    c = _started()
    c.abort()
    c.abort()
    assert c.phase == SessionPhase.ABORTED


def test_update_after_aborted_returns_empty():
    c = _started()
    c.update(COUNT_IN_SAMPLE)
    c.abort()
    assert c.update(PAST_ALL_SAMPLE) == []


# ── Summary ───────────────────────────────────────────────────────────────────

def test_summary_returns_none_during_session():
    c = _started()
    c.update(COUNT_IN_SAMPLE)
    assert c.phase == SessionPhase.ACTIVE
    assert c.summary() is None


def test_summary_returns_metrics_after_complete():
    c = _started()
    c.update(PAST_ALL_SAMPLE)
    m = c.summary()
    assert isinstance(m, LogMetrics)
    assert m.targets_total == 2
    assert m.targets_missed == 2
    assert m.targets_hit == 0


def test_summary_reflects_hits():
    c = _started()
    c.update(round(1.1 * SR), [NOM_0])
    c.update(round(1.6 * SR), [NOM_1])
    m = c.summary()
    assert m.targets_hit == 2
    assert m.targets_missed == 0
    assert m.good_hits == 2


def test_summary_returns_metrics_after_aborted_with_events():
    c = _started()
    # Log one miss before aborting
    c.update(round(DL_0 * SR) + 1)
    c.abort()
    m = c.summary()
    assert isinstance(m, LogMetrics)
    assert m.targets_missed >= 1


def test_summary_returns_none_before_start():
    c = _make()
    assert c.summary() is None


# ── Feedback event ordering and content ──────────────────────────────────────

def test_miss_events_in_ascending_target_order():
    c = _started()
    events = c.update(PAST_ALL_SAMPLE)
    indices = [ev["target_idx"] for ev in events]
    assert indices == sorted(indices)
    assert indices == [0, 1]


def test_hit_event_has_required_fields():
    c = _started()
    events = c.update(round(1.1 * SR), [NOM_0])
    assert len(events) == 1
    ev = events[0]
    assert "severity" in ev
    assert "messages" in ev
    assert "timing_error_s" in ev
    assert "target_idx" in ev
    assert ev["target_idx"] == 0


def test_exact_hit_severity_is_good():
    c = _started()
    events = c.update(round(1.1 * SR), [NOM_0])
    assert events[0]["severity"] == "good"
    assert events[0]["timing_error_s"] == pytest.approx(0.0, abs=1e-9)


def test_miss_event_severity_is_miss():
    c = _started()
    events = c.update(PAST_ALL_SAMPLE)
    assert all(ev["severity"] == "miss" for ev in events)


# ── Onset timing edge cases ───────────────────────────────────────────────────

def test_onset_before_target_window_ignored():
    """An onset well before count-in should not match any target."""
    c = _started()
    # Transition to ACTIVE, pass an onset at t=0.1s (before any target window)
    events = c.update(COUNT_IN_SAMPLE, [0.1])
    assert events == []
    assert c.phase == SessionPhase.ACTIVE


def test_onset_passed_at_transition_tick_is_processed():
    """Onsets passed on the same update() call that transitions to ACTIVE are matched."""
    c = _started()
    # Pass an onset at exactly the nominal time for target 0 on the transition tick
    events = c.update(COUNT_IN_SAMPLE, [NOM_0])
    assert len(events) == 1
    assert events[0]["target_idx"] == 0
    assert events[0]["severity"] == "good"
