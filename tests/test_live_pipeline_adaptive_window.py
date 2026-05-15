"""Tests for conservative adaptive extraction-window positioning in live_pipeline.py.

All tests use synthetic audio — no hardware involved.

The window-positioning feature:
    When a TempoTracker is active, the extraction window center is shifted
    from the nominal beat time toward the tracker's adjusted_target_time by
    a fraction (adaptive_window_shift, default 0.5), subject to a maximum
    shift (max_window_shift_s, default 0.30 × beat_s).

Test matrix
-----------
1.  Fixed-grid mode: window center equals nominal beat time.
2.  Adaptive mode before first observation: window shift is zero (tracker has no anchor).
3.  Moderate drift: window center is strictly between nominal and adjusted target.
4.  Moderate drift: window shift equals shift_fraction × (adjusted − nominal).
5.  Large drift: window shift is clamped to ±max_window_shift_s.
6.  Custom shift fraction: window moves the configured fraction toward adjusted.
7.  Custom max shift: clamping uses the supplied cap, not the default.
8.  Missed onset: tracker state unchanged; next window center is unaffected.
9.  End-to-end: onset captured in shifted window that nominal window would miss.
10. window_center_time_s event field is always nominal when shift is zero.
11. window_shift_s event field is zero in fixed mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_pipeline import process_realtime_audio, _compute_adaptive_window
from core.practice_session import PracticeSession
from core.tempo_tracker import TempoTracker


# ── Shared constants ──────────────────────────────────────────────────────────

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000
_BEAT_S   = 60.0 / _BPM       # 0.5 s
_CI_S     = _COUNT_IN * _BEAT_S  # 2.0 s
# Default max shift at _BPM
_MAX_SHIFT_DEFAULT = 0.30 * _BEAT_S  # 0.15 s

_SILENCE = np.zeros(500_000)


def _session(n: int = 10):
    return PracticeSession([{"time": i} for i in range(n)], _BPM, _COUNT_IN, _SR)


def _threshold_sample(beat_idx: int) -> int:
    """One sample past the ready threshold for beat_idx at _BPM / _COUNT_IN."""
    return round((_CI_S + beat_idx * _BEAT_S + 0.45) * _SR) + 1


def _pretrained_tracker(player_bpm: float, n_obs: int = 8) -> TempoTracker:
    """Return a TempoTracker pre-fed with n_obs observations at player_bpm."""
    tracker = TempoTracker(_BPM)
    player_beat_s = 60.0 / player_bpm
    for i in range(n_obs):
        nom = _CI_S + i * _BEAT_S
        act = _CI_S + i * player_beat_s
        tracker.observe(nom, act)
    return tracker


def _spike_audio(n_beats: int, player_bpm: float, n_samples: int = 500_000) -> np.ndarray:
    audio = np.zeros(n_samples)
    player_beat_s = 60.0 / player_bpm
    for i in range(n_beats):
        idx = round((_CI_S + i * player_beat_s) * _SR)
        if 0 <= idx < n_samples:
            audio[idx] = 0.8
    return audio


# ── 1. Fixed-grid: window center stays at nominal ─────────────────────────────

def test_fixed_mode_no_window_shift_field():
    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s)
    assert "window_shift_s" not in events[0]


def test_fixed_mode_no_window_center_field():
    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s)
    assert "window_center_time_s" not in events[0]


# ── 2. Adaptive before first observation: zero shift ─────────────────────────

def test_adaptive_no_anchor_zero_shift():
    """Tracker with no anchor: adjusted == nominal, shift is zero."""
    tracker = TempoTracker(_BPM)
    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s,
                                    tempo_tracker=tracker)
    assert events[0]["window_shift_s"] == pytest.approx(0.0, abs=1e-9)


def test_adaptive_no_anchor_center_equals_nominal():
    tracker = TempoTracker(_BPM)
    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s,
                                    tempo_tracker=tracker)
    ev = events[0]
    assert ev["window_center_time_s"] == pytest.approx(ev["nominal_target_time_s"], abs=1e-9)


# ── 3 & 4. Moderate drift: partial shift toward adjusted ─────────────────────

def test_moderate_drift_window_center_between_nominal_and_adjusted():
    """Window center should lie strictly between nominal and adjusted target."""
    tracker = _pretrained_tracker(player_bpm=121.0)

    # Beat 8 (not in the training set)
    nom_8 = _CI_S + 8 * _BEAT_S
    adj_8 = tracker.adjusted_target_time(nom_8)
    assert adj_8 < nom_8, "121 BPM player should be tracked as earlier than nominal"

    s = _session(10)
    s.evaluated_indices = set(range(8))
    events = process_realtime_audio(_SILENCE, _threshold_sample(8), s,
                                    tempo_tracker=tracker)
    ctr = events[0]["window_center_time_s"]
    assert adj_8 < ctr < nom_8, (
        f"Window center {ctr:.6f} should be in ({adj_8:.6f}, {nom_8:.6f})"
    )


def test_moderate_drift_shift_equals_fraction_times_gap():
    """window_shift_s should equal shift_fraction × (adjusted − nominal)."""
    tracker = _pretrained_tracker(player_bpm=121.0)
    nom_8 = _CI_S + 8 * _BEAT_S
    adj_8 = tracker.adjusted_target_time(nom_8)
    expected_raw_shift = 0.5 * (adj_8 - nom_8)

    # Only relevant when the raw shift is within the clamp
    assert abs(expected_raw_shift) < _MAX_SHIFT_DEFAULT, \
        "Pre-condition: shift should not be clamped for 121 BPM moderate drift"

    s = _session(10)
    s.evaluated_indices = set(range(8))
    events = process_realtime_audio(_SILENCE, _threshold_sample(8), s,
                                    tempo_tracker=tracker)
    assert events[0]["window_shift_s"] == pytest.approx(expected_raw_shift, abs=1e-9)


def test_moderate_drift_window_center_plus_shift_equals_nominal():
    """window_center_time_s == nominal + window_shift_s."""
    tracker = _pretrained_tracker(player_bpm=119.5)
    s = _session(10)
    s.evaluated_indices = set(range(8))
    events = process_realtime_audio(_SILENCE, _threshold_sample(8), s,
                                    tempo_tracker=tracker)
    ev = events[0]
    assert ev["window_center_time_s"] == pytest.approx(
        ev["nominal_target_time_s"] + ev["window_shift_s"], abs=1e-9
    )


# ── 5. Large drift: clamped to max shift ─────────────────────────────────────

def test_large_drift_shift_clamped_to_default_max():
    """When raw shift exceeds the 30%-beat cap, window_shift_s is clamped."""
    # 60 BPM: beat_s = 1.0 s, default max_shift = 0.30 s
    nominal_bpm = 60.0
    beat_s_60   = 1.0
    max_shift   = 0.30 * beat_s_60

    tracker = TempoTracker(nominal_bpm)
    count_in_s = 2 * beat_s_60
    # Pre-adapt heavily: 30 observations at 70 BPM (much faster)
    player_beat_s = 60.0 / 70.0
    for i in range(30):
        tracker.observe(count_in_s + i * beat_s_60,
                        count_in_s + i * player_beat_s)

    # Verify that the raw shift would exceed the cap
    nom_30 = count_in_s + 30 * beat_s_60
    adj_30 = tracker.adjusted_target_time(nom_30)
    raw_shift = 0.5 * (adj_30 - nom_30)
    assert abs(raw_shift) > max_shift, "Pre-condition: raw shift must exceed cap"

    sr = 48000
    session = PracticeSession([{"time": i} for i in range(31)], nominal_bpm, 2, sr)
    session.evaluated_indices = set(range(30))

    threshold_s    = nom_30 + 0.35 + 0.15
    current_sample = round(threshold_s * sr) + 1

    events = process_realtime_audio(
        np.zeros(2_500_000), current_sample, session, tempo_tracker=tracker
    )
    assert len(events) == 1
    actual_shift = events[0]["window_shift_s"]
    # Should be clamped at -max_shift (player early → negative shift)
    assert actual_shift == pytest.approx(-max_shift, abs=1e-6)


def test_large_drift_center_is_nominal_plus_clamped_shift():
    nominal_bpm   = 60.0
    beat_s_60     = 1.0
    max_shift     = 0.30 * beat_s_60
    tracker       = TempoTracker(nominal_bpm)
    count_in_s    = 2 * beat_s_60
    player_beat_s = 60.0 / 70.0
    for i in range(30):
        tracker.observe(count_in_s + i * beat_s_60,
                        count_in_s + i * player_beat_s)

    sr         = 48000
    nom_30     = count_in_s + 30 * beat_s_60
    session    = PracticeSession([{"time": i} for i in range(31)], nominal_bpm, 2, sr)
    session.evaluated_indices = set(range(30))
    threshold_s    = nom_30 + 0.35 + 0.15
    current_sample = round(threshold_s * sr) + 1

    events = process_realtime_audio(
        np.zeros(2_500_000), current_sample, session, tempo_tracker=tracker
    )
    ev = events[0]
    assert ev["window_center_time_s"] == pytest.approx(
        ev["nominal_target_time_s"] + ev["window_shift_s"], abs=1e-9
    )


# ── 6. Custom shift fraction ──────────────────────────────────────────────────

def test_custom_shift_fraction_applied():
    """adaptive_window_shift=0.25 moves the window only a quarter of the gap."""
    tracker = _pretrained_tracker(player_bpm=121.0)
    nom_8 = _CI_S + 8 * _BEAT_S
    adj_8 = tracker.adjusted_target_time(nom_8)

    s = _session(10)
    s.evaluated_indices = set(range(8))
    events = process_realtime_audio(
        _SILENCE, _threshold_sample(8), s,
        tempo_tracker=tracker, adaptive_window_shift=0.25,
    )
    expected_shift = 0.25 * (adj_8 - nom_8)
    assert events[0]["window_shift_s"] == pytest.approx(expected_shift, abs=1e-9)


def test_zero_shift_fraction_no_window_movement():
    """adaptive_window_shift=0.0 keeps the window at the nominal position."""
    tracker = _pretrained_tracker(player_bpm=121.0)
    s = _session(10)
    s.evaluated_indices = set(range(8))
    events = process_realtime_audio(
        _SILENCE, _threshold_sample(8), s,
        tempo_tracker=tracker, adaptive_window_shift=0.0,
    )
    ev = events[0]
    assert ev["window_shift_s"] == pytest.approx(0.0, abs=1e-9)
    assert ev["window_center_time_s"] == pytest.approx(ev["nominal_target_time_s"], abs=1e-9)


# ── 7. Custom max shift ───────────────────────────────────────────────────────

def test_custom_max_shift_respected():
    """max_window_shift_s is honoured when a tight cap is supplied."""
    tracker = _pretrained_tracker(player_bpm=121.0)
    nom_8 = _CI_S + 8 * _BEAT_S
    adj_8 = tracker.adjusted_target_time(nom_8)
    raw_shift = 0.5 * (adj_8 - nom_8)

    cap = 0.002  # 2 ms — much smaller than the raw shift
    assert abs(raw_shift) > cap, "Pre-condition: raw shift should exceed tiny cap"

    s = _session(10)
    s.evaluated_indices = set(range(8))
    events = process_realtime_audio(
        _SILENCE, _threshold_sample(8), s,
        tempo_tracker=tracker, max_window_shift_s=cap,
    )
    assert abs(events[0]["window_shift_s"]) == pytest.approx(cap, abs=1e-9)


# ── 8. Missed onset does not affect subsequent window positions ───────────────

def test_missed_onset_does_not_move_later_window():
    """
    When no onset is found for a beat, the tracker is not updated.
    The window center for the next beat should equal what it would have been
    had the missed beat never been presented.
    """
    tracker = _pretrained_tracker(player_bpm=121.0, n_obs=6)

    # Compute expected window center for beat 7 *before* processing beat 6
    nom_7 = _CI_S + 7 * _BEAT_S
    adj_7 = tracker.adjusted_target_time(nom_7)
    raw_7 = 0.5 * (adj_7 - nom_7)
    expected_shift_7 = max(-_MAX_SHIFT_DEFAULT, min(_MAX_SHIFT_DEFAULT, raw_7))
    expected_center_7 = nom_7 + expected_shift_7

    # Process beat 6 silently (miss)
    s = _session(9)
    s.evaluated_indices = set(range(6))
    process_realtime_audio(_SILENCE, _threshold_sample(6), s, tempo_tracker=tracker)

    # Process beat 7 and verify window center is unchanged
    events_7 = process_realtime_audio(_SILENCE, _threshold_sample(7), s, tempo_tracker=tracker)
    assert len(events_7) == 1
    assert events_7[0]["window_center_time_s"] == pytest.approx(expected_center_7, abs=1e-9)


# ── 9. End-to-end: shifted window captures onset that nominal would miss ──────

def test_onset_captured_in_shifted_window_missed_by_nominal():
    """
    At 121 BPM, beat 8's onset lands ~3 ms before the nominal window's left edge.
    After adapting to 8 beats of 121 BPM play, the shifted window moves earlier
    and captures the onset.

    Verified analytically:
        player_beat_s ≈ 0.49587 s
        beat 8 actual ≈ 2.0 + 8 × 0.49587 = 5.9670 s
        nominal window start = 2.0 + 8 × 0.5 − 0.03 = 5.970 s  (MISSES onset)
        shifted window start (≈ shift −5 ms) ≈ 5.965 s           (CAPTURES onset)
    """
    player_bpm    = 121.0
    player_beat_s = 60.0 / player_bpm
    n_train       = 8

    # Confirm the geometry (regression guard)
    beat_8_actual = _CI_S + 8 * player_beat_s
    nom_window_start = _CI_S + 8 * _BEAT_S - 0.03
    assert beat_8_actual < nom_window_start, (
        f"Pre-condition failed: beat 8 ({beat_8_actual:.5f} s) should be "
        f"before the nominal window start ({nom_window_start:.5f} s)"
    )

    audio = _spike_audio(n_train + 1, player_bpm)

    # ── Fixed-grid: should NOT detect beat 8 ─────────────────────────────────
    session_fixed = _session(n_train + 1)
    for i in range(n_train):
        process_realtime_audio(audio, _threshold_sample(i), session_fixed)
    evs_fixed = process_realtime_audio(audio, _threshold_sample(n_train), session_fixed)

    assert len(evs_fixed) == 1
    assert evs_fixed[0]["detected_note"] is None, (
        "Fixed grid should miss the onset that lands before its window"
    )

    # ── Adaptive: should detect beat 8 via shifted window ────────────────────
    session_adaptive = _session(n_train + 1)
    tracker = TempoTracker(_BPM)
    for i in range(n_train):
        process_realtime_audio(audio, _threshold_sample(i), session_adaptive,
                               tempo_tracker=tracker)
    evs_adaptive = process_realtime_audio(audio, _threshold_sample(n_train),
                                          session_adaptive, tempo_tracker=tracker)

    assert len(evs_adaptive) == 1
    assert evs_adaptive[0]["detected_note"] == "?", (
        "Adaptive grid should detect the onset via its shifted window"
    )

    # The window center should have moved earlier than nominal
    assert evs_adaptive[0]["window_shift_s"] < 0, "Shift should be negative (earlier)"
    assert evs_adaptive[0]["window_center_time_s"] < evs_adaptive[0]["nominal_target_time_s"]


# ── 10 & 11. _compute_adaptive_window unit tests ──────────────────────────────

def test_compute_adaptive_window_no_anchor():
    """With no anchor, adjusted == nominal, shift is zero."""
    tracker = TempoTracker(60.0)
    center, shift = _compute_adaptive_window(5.0, tracker, 0.5, 0.30)
    assert shift  == pytest.approx(0.0, abs=1e-9)
    assert center == pytest.approx(5.0, abs=1e-9)


def test_compute_adaptive_window_partial_shift():
    """shift_fraction=0.5 moves center halfway between nominal and adjusted."""
    tracker = TempoTracker(60.0)
    tracker.observe(2.0, 2.0)   # anchor
    tracker.observe(3.0, 2.95)  # player 50 ms early → tempo_ratio < 1
    nom = 4.0
    adj = tracker.adjusted_target_time(nom)
    center, shift = _compute_adaptive_window(nom, tracker, 0.5, 1.0)  # big cap
    assert shift  == pytest.approx(0.5 * (adj - nom), abs=1e-9)
    assert center == pytest.approx(nom + shift, abs=1e-9)


def test_compute_adaptive_window_clamp_positive():
    """Positive raw shift is clamped at +max_shift_s."""
    tracker = TempoTracker(60.0)
    tracker.observe(2.0, 2.0)
    tracker.observe(3.0, 3.10)  # player 100 ms late → adj > nom
    nom    = 4.0
    center, shift = _compute_adaptive_window(nom, tracker, 1.0, 0.02)
    assert shift  == pytest.approx(0.02, abs=1e-9)
    assert center == pytest.approx(nom + 0.02, abs=1e-9)


def test_compute_adaptive_window_clamp_negative():
    """Negative raw shift is clamped at -max_shift_s."""
    tracker = TempoTracker(60.0)
    tracker.observe(2.0, 2.0)
    tracker.observe(3.0, 2.90)  # player 100 ms early → adj < nom
    nom = 4.0
    center, shift = _compute_adaptive_window(nom, tracker, 1.0, 0.02)
    assert shift  == pytest.approx(-0.02, abs=1e-9)
    assert center == pytest.approx(nom - 0.02, abs=1e-9)
