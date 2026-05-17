"""Tests for core/metrics.py.

All tests are pure Python — no audio hardware required.

Test matrix
-----------
Empty input
  1.  Empty list → all-zero SessionMetrics, timing=None, pitch=None.

All-good session
  2.  N detected "good" events → hit_rate=1.0, good_count=N.
  3.  Timing stats values match manual calculation.
  4.  std_error_s == 0.0 when all timing errors are identical.

All-miss session (no-shows)
  5.  N no-show events → detected=0, hit_rate=0.0, timing=None.
  6.  missed_indices contains all target_idx values in event order.

Mixed session
  7.  Mix of good, warn, no-show → counts match input.
  8.  late_hit_count counts detected events with severity=="miss".
  9.  good_rate / warn_rate exclude no-shows from denominator.

Timing statistics
  10. mean_error_s is signed positive for an all-late session.
  11. mean_error_s is signed negative for an all-early session.
  12. std_error_s correct for a known two-point sequence.
  13. max_abs_error_s is the largest |timing_error_s|.
  14. No-show events (timing_error_s=None) excluded from timing stats.

Streak analysis
  15. Longest good streak: G G ∅ G G G → 3.
  16. Longest miss streak: G ∅ ∅ G ∅ → 2.
  17. Longest clean streak (detected, any severity): D D ∅ D D → 2.
  18. current_good_streak: session ending on a no-show → 0.
  19. current_good_streak: session ending on 3 consecutive goods → 3.

Missed indices
  20. missed_indices contains exactly the target_idx of no-show events.
  21. missed_indices is a tuple.

Pitch stats
  22. pitch=None when all pitch_error_cents are None.
  23. in_tune_rate correct for a mix of in-tune / out-of-tune events.

Rolling metrics
  24. len(result) == max(0, len(events) - window + 1).
  25. Window of all-good events → hit_rate=1.0, good_rate=1.0.
  26. Window spanning a no-show → hit_rate < 1.0.
  27. mean_abs_timing_s=None for a window with no detected events.
  28. window > len(events) → empty list.

Format
  29. format_session_metrics returns a non-empty string.
  30. format_session_metrics with bpm=None does not raise.
  31. format_session_metrics with bpm set includes beat annotation.

Integration (uses replay fixtures + replay_session_data)
  32. four_quarter_120bpm_all_hits → hit_rate=1.0, good_count=4.
  33. four_quarter_120bpm_one_miss → hit_rate=0.75, undetected=1.
  34. two_beats_100bpm_warn → hit_rate=1.0, warn_count=2, good_count=0.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import (
    SessionMetrics,
    StreakStats,
    TimingStats,
    compute_rolling_metrics,
    compute_session_metrics,
    format_session_metrics,
)
from core.session_replay import replay_session_data

# ── Event factories ───────────────────────────────────────────────────────────

def _hit(
    target_idx: int,
    timing_error_s: float,
    pitch_error_cents: float | None = None,
    severity: str | None = None,
) -> dict:
    """Detected-note event. Severity auto-computed from timing if not given."""
    if severity is None:
        abs_t = abs(timing_error_s)
        if abs_t <= 0.05:
            severity = "good"
        elif abs_t <= 0.12:
            severity = "warn"
        else:
            severity = "miss"
    return {
        "target_idx":          target_idx,
        "expected_note":       "?",
        "detected_note":       "?",
        "timing_error_s":      timing_error_s,
        "pitch_error_cents":   pitch_error_cents,
        "confidence":          None,
        "severity":            severity,
        "messages":            [],
    }


def _noshow(target_idx: int) -> dict:
    """No note detected — true miss."""
    return {
        "target_idx":          target_idx,
        "expected_note":       "?",
        "detected_note":       None,
        "timing_error_s":      None,
        "pitch_error_cents":   None,
        "confidence":          None,
        "severity":            "miss",
        "messages":            ["No note detected"],
    }


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sessions"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


# ── 1: Empty input ────────────────────────────────────────────────────────────

def test_empty_events_zero_metrics():
    m = compute_session_metrics([])
    assert m.total      == 0
    assert m.detected   == 0
    assert m.undetected == 0
    assert m.hit_rate   == pytest.approx(0.0)
    assert m.good_count == 0
    assert m.warn_count == 0
    assert m.miss_count == 0
    assert m.timing     is None
    assert m.pitch      is None
    assert m.missed_indices == ()
    assert m.streaks.longest_good_streak  == 0
    assert m.streaks.longest_miss_streak  == 0
    assert m.streaks.current_good_streak  == 0


# ── 2–4: All-good session ─────────────────────────────────────────────────────

def test_all_good_counts():
    events = [_hit(i, 0.01) for i in range(4)]
    m = compute_session_metrics(events)
    assert m.total      == 4
    assert m.detected   == 4
    assert m.undetected == 0
    assert m.hit_rate   == pytest.approx(1.0)
    assert m.good_count == 4
    assert m.warn_count == 0
    assert m.miss_count == 0


def test_timing_stats_values():
    # errors: +0.01, +0.02, +0.03 → mean=0.02, mean_abs=0.02
    events = [_hit(i, e) for i, e in enumerate([0.01, 0.02, 0.03])]
    m = compute_session_metrics(events)
    assert m.timing is not None
    assert m.timing.count            == 3
    assert m.timing.mean_error_s     == pytest.approx(0.02)
    assert m.timing.mean_abs_error_s == pytest.approx(0.02)
    assert m.timing.max_abs_error_s  == pytest.approx(0.03)
    expected_std = math.sqrt(((0.01 - 0.02)**2 + 0 + (0.03 - 0.02)**2) / 3)
    assert m.timing.std_error_s      == pytest.approx(expected_std)


def test_std_zero_when_all_errors_identical():
    events = [_hit(i, 0.01) for i in range(5)]
    m = compute_session_metrics(events)
    assert m.timing.std_error_s == pytest.approx(0.0)


# ── 5–6: All-miss (no-show) session ──────────────────────────────────────────

def test_all_noshows_zero_detection():
    events = [_noshow(i) for i in range(3)]
    m = compute_session_metrics(events)
    assert m.detected   == 0
    assert m.undetected == 3
    assert m.hit_rate   == pytest.approx(0.0)
    assert m.timing     is None
    assert m.good_rate  is None
    assert m.warn_rate  is None


def test_missed_indices_all_noshows():
    events = [_noshow(i) for i in [0, 2, 5]]
    m = compute_session_metrics(events)
    assert m.missed_indices == (0, 2, 5)


# ── 7–9: Mixed session ────────────────────────────────────────────────────────

def test_mixed_counts():
    events = [
        _hit(0, 0.01),       # good
        _hit(1, 0.08),       # warn
        _noshow(2),          # no-show
    ]
    m = compute_session_metrics(events)
    assert m.total      == 3
    assert m.detected   == 2
    assert m.undetected == 1
    assert m.good_count == 1
    assert m.warn_count == 1
    # severity miss_count includes the no-show
    assert m.miss_count == 1


def test_late_hit_count():
    # A detected note outside the 120 ms threshold gets severity "miss"
    late = _hit(0, 0.20, severity="miss")   # detected but very late
    noshow = _noshow(1)
    m = compute_session_metrics([late, noshow])
    # Two miss-severity events: one late hit + one no-show
    assert m.miss_count     == 2
    assert m.late_hit_count == 1   # only the detected one
    assert m.undetected     == 1   # only the no-show


def test_rates_exclude_noshows():
    events = [_hit(0, 0.01), _hit(1, 0.01), _noshow(2)]
    m = compute_session_metrics(events)
    # good_rate is over detected (2), not total (3)
    assert m.good_rate == pytest.approx(1.0)
    assert m.warn_rate == pytest.approx(0.0)


# ── 10–14: Timing statistics ──────────────────────────────────────────────────

def test_mean_error_signed_positive_for_late():
    events = [_hit(i, 0.05) for i in range(3)]   # all 50 ms late
    m = compute_session_metrics(events)
    assert m.timing.mean_error_s > 0


def test_mean_error_signed_negative_for_early():
    events = [_hit(i, -0.03) for i in range(3)]  # all 30 ms early
    m = compute_session_metrics(events)
    assert m.timing.mean_error_s < 0


def test_std_error_known_sequence():
    # errors [0.0, 0.1] → mean=0.05, population std=0.05
    events = [_hit(0, 0.0), _hit(1, 0.1)]
    m = compute_session_metrics(events)
    assert m.timing.std_error_s == pytest.approx(0.05)


def test_max_abs_error():
    events = [_hit(0, 0.01), _hit(1, -0.04), _hit(2, 0.03)]
    m = compute_session_metrics(events)
    assert m.timing.max_abs_error_s == pytest.approx(0.04)


def test_noshows_excluded_from_timing():
    events = [_hit(0, 0.01), _noshow(1), _hit(2, 0.03)]
    m = compute_session_metrics(events)
    assert m.timing is not None
    assert m.timing.count == 2   # only the two detected events


# ── 15–19: Streak analysis ────────────────────────────────────────────────────

def test_longest_good_streak():
    # G G ∅ G G G → longest good = 3
    events = [
        _hit(0, 0.01),   # good
        _hit(1, 0.01),   # good
        _noshow(2),
        _hit(3, 0.01),   # good
        _hit(4, 0.01),   # good
        _hit(5, 0.01),   # good
    ]
    m = compute_session_metrics(events)
    assert m.streaks.longest_good_streak == 3


def test_longest_miss_streak():
    # G ∅ ∅ G ∅ → longest miss = 2
    events = [
        _hit(0, 0.01),
        _noshow(1),
        _noshow(2),
        _hit(3, 0.01),
        _noshow(4),
    ]
    m = compute_session_metrics(events)
    assert m.streaks.longest_miss_streak == 2


def test_longest_clean_streak():
    # D D ∅ D D → longest clean = 2
    events = [
        _hit(0, 0.01),
        _hit(1, 0.08),   # warn but detected
        _noshow(2),
        _hit(3, 0.01),
        _hit(4, 0.08),
    ]
    m = compute_session_metrics(events)
    assert m.streaks.longest_clean_streak == 2


def test_current_good_streak_ends_on_miss():
    events = [_hit(0, 0.01), _hit(1, 0.01), _noshow(2)]
    m = compute_session_metrics(events)
    assert m.streaks.current_good_streak == 0


def test_current_good_streak_ends_on_goods():
    events = [_noshow(0), _hit(1, 0.01), _hit(2, 0.01), _hit(3, 0.01)]
    m = compute_session_metrics(events)
    assert m.streaks.current_good_streak == 3


# ── 20–21: Missed indices ─────────────────────────────────────────────────────

def test_missed_indices_preserves_event_order():
    # Miss at target 3 emitted after hit at target 4 (as in the replay harness)
    events = [_hit(0, 0.01), _hit(1, 0.01), _hit(4, 0.02), _noshow(3)]
    m = compute_session_metrics(events)
    assert m.missed_indices == (3,)


def test_missed_indices_is_tuple():
    m = compute_session_metrics([_noshow(0), _noshow(2)])
    assert isinstance(m.missed_indices, tuple)


# ── 22–23: Pitch stats ────────────────────────────────────────────────────────

def test_pitch_none_when_no_pitch_data():
    events = [_hit(0, 0.01), _hit(1, 0.02)]  # pitch_error_cents=None
    m = compute_session_metrics(events)
    assert m.pitch is None


def test_pitch_in_tune_rate():
    # 25 cents = exactly on the boundary (≤ 25 → in tune)
    events = [
        _hit(0, 0.01, pitch_error_cents=10.0),   # in tune
        _hit(1, 0.01, pitch_error_cents=25.0),   # on boundary → in tune
        _hit(2, 0.01, pitch_error_cents=-30.0),  # out of tune
        _hit(3, 0.01, pitch_error_cents=50.0),   # out of tune
    ]
    m = compute_session_metrics(events)
    assert m.pitch is not None
    assert m.pitch.count == 4
    assert m.pitch.in_tune_rate == pytest.approx(0.5)   # 2 of 4 in tune
    assert m.pitch.mean_error_cents == pytest.approx((10 + 25 - 30 + 50) / 4)


# ── 24–28: Rolling metrics ────────────────────────────────────────────────────

def test_rolling_length():
    events = [_hit(i, 0.01) for i in range(10)]
    for w in (1, 3, 5, 10):
        result = compute_rolling_metrics(events, window=w)
        assert len(result) == max(0, len(events) - w + 1)


def test_rolling_all_good_window():
    events = [_hit(i, 0.01) for i in range(4)]
    result = compute_rolling_metrics(events, window=4)
    assert len(result) == 1
    assert result[0].hit_rate   == pytest.approx(1.0)
    assert result[0].good_rate  == pytest.approx(1.0)
    assert result[0].end_idx    == 3


def test_rolling_window_spanning_noshow():
    events = [_hit(0, 0.01), _noshow(1), _hit(2, 0.01)]
    result = compute_rolling_metrics(events, window=3)
    assert len(result) == 1
    assert result[0].hit_rate == pytest.approx(2 / 3)


def test_rolling_mean_timing_none_for_all_noshows():
    events = [_noshow(0), _noshow(1), _noshow(2)]
    result = compute_rolling_metrics(events, window=3)
    assert len(result) == 1
    assert result[0].mean_abs_timing_s is None
    assert result[0].good_rate         is None


def test_rolling_window_larger_than_events():
    events = [_hit(i, 0.01) for i in range(3)]
    assert compute_rolling_metrics(events, window=10) == []


# ── 29–31: Format ─────────────────────────────────────────────────────────────

def test_format_returns_nonempty_string():
    m = compute_session_metrics([_hit(0, 0.01)])
    assert len(format_session_metrics(m)) > 0


def test_format_bpm_none_does_not_raise():
    m = compute_session_metrics([_hit(0, 0.01), _noshow(1)])
    format_session_metrics(m, bpm=None)  # must not raise


def test_format_bpm_set_includes_beats():
    events = [_hit(i, 0.01) for i in range(4)]
    m = compute_session_metrics(events)
    output = format_session_metrics(m, bpm=120.0)
    assert "beat" in output.lower()


# ── 32–34: Integration with replay fixtures ───────────────────────────────────

def test_integration_all_hits_fixture():
    data   = _load_fixture("four_quarter_120bpm_all_hits.json")
    events = replay_session_data(data)
    m      = compute_session_metrics(events)
    assert m.hit_rate   == pytest.approx(1.0)
    assert m.good_count == 4
    assert m.warn_count == 0
    assert m.undetected == 0
    assert m.timing is not None


def test_integration_one_miss_fixture():
    data   = _load_fixture("four_quarter_120bpm_one_miss.json")
    events = replay_session_data(data)
    m      = compute_session_metrics(events)
    assert m.hit_rate   == pytest.approx(0.75)
    assert m.undetected == 1
    assert m.detected   == 3
    assert len(m.missed_indices) == 1


def test_integration_warn_fixture():
    data   = _load_fixture("two_beats_100bpm_warn.json")
    events = replay_session_data(data)
    m      = compute_session_metrics(events)
    assert m.hit_rate   == pytest.approx(1.0)
    assert m.warn_count == 2
    assert m.good_count == 0
    assert m.undetected == 0
