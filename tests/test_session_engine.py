"""Tests for core/session_engine.py.

All tests are synthetic — no audio hardware required.

Test matrix
-----------
Matching (on_onset)
    1.  Exact hit: onset at nominal → timing_error_s == 0, severity != "miss".
    2.  Early hit (within window) → negative timing_error_s, severity "good".
    3.  Late hit (within window, ≤ 50 ms) → positive timing_error_s, severity "good".
    4.  Late hit (within window, ~ 100 ms) → severity "warn".
    5.  Onset before count-in ends → unmatched, [].
    6.  Onset beyond all target windows → unmatched, [].
    7.  Custom (narrow) match_window_s excludes onset that default would accept.
    8.  Custom (wide) match_window_s accepts onset that default would reject.
    9.  Already-evaluated target: second onset → [].
    10. evaluated_indices updated after a hit.
    11. Equidistant tie between two targets → lower index wins.

Missed targets (update_time)
    12. Time past deadline, no onset → miss event emitted.
    13. update_time called twice past deadline → miss emitted exactly once.
    14. Time before deadline → [].
    15. Hit by on_onset then update_time past deadline → no miss for that target.
    16. evaluated_indices updated after a miss.

Sequential / integration
    17. Two onsets in order match the two correct targets.
    18. Mixed hit + miss: target 0 hit, target 1 missed by update_time.

TempoTracker integration
    19. Matched onset updates tracker (tracker.has_anchor becomes True).
    20. Unmatched onset does NOT update tracker.
    21. tracker=None is safe (no AttributeError).

Event format
    22. Hit event contains all fields from feedback_event().
    23. Miss event contains all fields from feedback_event().
    24. Hit event includes extra "onset_time_s" field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_engine import SessionEngine
from core.tempo_tracker import TempoTracker

# ── Shared constants ──────────────────────────────────────────────────────────

NOM_BPM    = 120.0
COUNT_IN   = 2
BEAT_S     = 60.0 / NOM_BPM      # 0.5 s
COUNT_IN_S = COUNT_IN * BEAT_S    # 1.0 s
HALF_BEAT  = BEAT_S * 0.5        # 0.25 s  (default match_window_s)

TARGETS = [
    {"time": 0},  # nominal = 1.0 s
    {"time": 1},  # nominal = 1.5 s
    {"time": 2},  # nominal = 2.0 s
    {"time": 3},  # nominal = 2.5 s
]

# Precomputed nominal absolute times.
NOM = [COUNT_IN_S + t["time"] * BEAT_S for t in TARGETS]
# NOM == [1.0, 1.5, 2.0, 2.5]

_FEEDBACK_FIELDS = (
    "target_idx", "severity", "timing_error_s", "detected_note",
    "pitch_error_cents", "confidence", "messages", "expected_note",
)


def _engine(**kwargs) -> SessionEngine:
    return SessionEngine(TARGETS, NOM_BPM, COUNT_IN, **kwargs)


# ── 1–4: Basic matching ───────────────────────────────────────────────────────

def test_exact_hit():
    eng = _engine()
    events = eng.on_onset(NOM[0])
    assert len(events) == 1
    ev = events[0]
    assert ev["target_idx"] == 0
    assert ev["timing_error_s"] == pytest.approx(0.0, abs=1e-9)
    assert ev["severity"] != "miss"


def test_early_hit():
    # Use 40 ms rather than the exact 50 ms threshold to avoid IEEE-754 boundary issues.
    offset = -0.040
    ev = _engine().on_onset(NOM[0] + offset)[0]
    assert ev["timing_error_s"] == pytest.approx(offset, abs=1e-9)
    assert ev["severity"] == "good"


def test_late_hit_good():
    offset = +0.040
    ev = _engine().on_onset(NOM[0] + offset)[0]
    assert ev["timing_error_s"] == pytest.approx(offset, abs=1e-9)
    assert ev["severity"] == "good"


def test_late_hit_warn():
    """100 ms late is within the default window (250 ms) but severity = warn."""
    offset = +0.100
    ev = _engine().on_onset(NOM[0] + offset)[0]
    assert ev["severity"] == "warn"


# ── 5–6: Unmatched onsets ────────────────────────────────────────────────────

def test_onset_before_count_in_is_unmatched():
    events = _engine().on_onset(0.0)
    assert events == []


def test_onset_beyond_all_windows_is_unmatched():
    """Past the last target's window by more than match_window_s."""
    beyond = NOM[-1] + HALF_BEAT + 0.01
    events = _engine().on_onset(beyond)
    assert events == []


# ── 7–8: Custom match_window_s ───────────────────────────────────────────────

def test_custom_narrow_window_rejects_onset():
    """Onset 100 ms late: rejected by a 50 ms window, would be accepted by default."""
    eng = _engine(match_window_s=0.050)
    events = eng.on_onset(NOM[0] + 0.100)
    assert events == []


def test_custom_wide_window_accepts_onset():
    """Onset 350 ms late: rejected by default half-beat (250 ms), accepted by wide window."""
    eng = _engine(match_window_s=0.400)
    events = eng.on_onset(NOM[0] + 0.350)
    assert len(events) == 1


# ── 9–10: Already-evaluated guard ────────────────────────────────────────────

def test_already_evaluated_target_ignored():
    eng = _engine()
    eng.on_onset(NOM[0])
    events = eng.on_onset(NOM[0] + 0.01)
    assert events == []


def test_evaluated_indices_updated_on_hit():
    eng = _engine()
    eng.on_onset(NOM[0])
    assert 0 in eng.evaluated_indices


# ── 11: Tie-breaking ─────────────────────────────────────────────────────────

def test_equidistant_tie_picks_lower_index():
    """Onset exactly halfway between target 0 and target 1 → target 0 wins."""
    eng = _engine()
    midpoint = (NOM[0] + NOM[1]) / 2.0  # 1.25 s; 0.25 s from each
    events = eng.on_onset(midpoint)
    assert len(events) == 1
    assert events[0]["target_idx"] == 0


# ── 12–16: Missed targets (update_time) ──────────────────────────────────────

def test_miss_event_emitted_past_deadline():
    eng = _engine()
    deadline = NOM[0] + HALF_BEAT
    events = eng.update_time(deadline + 0.001)
    assert len(events) == 1
    ev = events[0]
    assert ev["target_idx"] == 0
    assert ev["severity"] == "miss"
    assert ev["timing_error_s"] is None
    assert ev["detected_note"] is None


def test_no_duplicate_miss_on_second_call():
    """Two update_time calls past the same deadline → miss emitted only once."""
    eng = _engine()
    deadline = NOM[0] + HALF_BEAT + 0.001
    eng.update_time(deadline)
    events = eng.update_time(deadline + 1.0)
    missed_indices = [ev["target_idx"] for ev in events]
    assert 0 not in missed_indices


def test_before_deadline_no_miss():
    events = _engine().update_time(NOM[0] - 0.001)
    assert events == []


def test_hit_prevents_later_miss():
    """Target hit by on_onset → update_time past its deadline emits no miss for it."""
    eng = _engine()
    eng.on_onset(NOM[0])
    deadline = NOM[0] + HALF_BEAT + 0.001
    events = eng.update_time(deadline)
    missed_indices = [ev["target_idx"] for ev in events]
    assert 0 not in missed_indices


def test_evaluated_indices_updated_on_miss():
    eng = _engine()
    eng.update_time(NOM[0] + HALF_BEAT + 0.001)
    assert 0 in eng.evaluated_indices


# ── 17–18: Sequential / integration ─────────────────────────────────────────

def test_two_targets_in_sequence():
    eng = _engine()
    ev0 = eng.on_onset(NOM[0] + 0.020)
    ev1 = eng.on_onset(NOM[1] - 0.030)
    assert len(ev0) == 1 and ev0[0]["target_idx"] == 0
    assert len(ev1) == 1 and ev1[0]["target_idx"] == 1
    assert ev0[0]["timing_error_s"] == pytest.approx(+0.020, abs=1e-9)
    assert ev1[0]["timing_error_s"] == pytest.approx(-0.030, abs=1e-9)


def test_mixed_hit_and_miss():
    """Target 0 hit; target 1's window closes before any onset → miss for 1 only."""
    eng = _engine()
    eng.on_onset(NOM[0])
    deadline_1 = NOM[1] + HALF_BEAT + 0.001
    events = eng.update_time(deadline_1)
    missed_indices = [ev["target_idx"] for ev in events]
    assert 0 not in missed_indices
    assert 1 in missed_indices


# ── 19–21: TempoTracker integration ─────────────────────────────────────────

def test_tracker_updated_on_matched_onset():
    tracker = TempoTracker(NOM_BPM)
    eng = _engine(tracker=tracker)
    assert not tracker.has_anchor
    eng.on_onset(NOM[0])
    assert tracker.has_anchor


def test_tracker_not_updated_on_unmatched_onset():
    tracker = TempoTracker(NOM_BPM)
    eng = _engine(tracker=tracker)
    eng.on_onset(0.0)   # before count-in — unmatched
    assert not tracker.has_anchor


def test_no_tracker_does_not_raise():
    eng = _engine(tracker=None)
    events = eng.on_onset(NOM[0])
    assert len(events) == 1


# ── 22–24: Event format ───────────────────────────────────────────────────────

def test_hit_event_has_all_feedback_fields():
    ev = _engine().on_onset(NOM[0])[0]
    for f in _FEEDBACK_FIELDS:
        assert f in ev, f"missing field: {f}"


def test_miss_event_has_all_feedback_fields():
    ev = _engine().update_time(NOM[0] + HALF_BEAT + 0.001)[0]
    for f in _FEEDBACK_FIELDS:
        assert f in ev, f"missing field: {f}"


def test_hit_event_includes_onset_time_s():
    onset = NOM[0] + 0.030
    ev = _engine().on_onset(onset)[0]
    assert "onset_time_s" in ev
    assert ev["onset_time_s"] == pytest.approx(onset, abs=1e-9)
