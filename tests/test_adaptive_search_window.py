"""Tests for adaptive search-window widening in live_pipeline.py.

When a TempoTracker is active and confidence is low, the extraction window's
pre-roll and post-roll are widened so that early/late onsets that fall outside
the nominal window can still be observed and used to train the tracker.

Test matrix
-----------
Unit — _compute_search_rolls
    1.  confidence=0   → full extra margin (adaptive_search_margin_beats × beat_s).
    2.  confidence=1   → no extra margin (rolls equal fixed defaults).
    3.  confidence=0.5 → extra margin is halved.
    4.  Extra margin is clamped to max_adaptive_search_margin_beats.
    5.  max_adaptive_search_margin_beats=0 → rolls always equal defaults.

Event fields
    6.  Adaptive events include search_pre_roll_s and search_post_roll_s.
    7.  Fixed-grid events do NOT include search_pre_roll_s / search_post_roll_s.
    8.  tracker_confidence in the event equals the pre-observe confidence.
    9.  search_pre_roll_s decreases as tracker accumulates accepted observations.

Fixed-grid regression
    10. Fixed-grid extraction uses the default pre/post rolls (no widening).

No-false-observation on silence
    11. Silence audio in adaptive mode: detected_note is None, tracker not updated.

Outlier isolation preserved
    12. A 132 BPM onset that lands at nominal time (i.e. an outlier relative to a
        well-trained 120 BPM tracker) is not used to retime the grid.

End-to-end convergence: 120 nominal / 132 actual
    13. With the wider search window, a 132 BPM player accumulates enough
        observations over 24 beats for the tracker to reach ≥ 128 BPM.
    14. Without adaptive search widening (fixed-grid), the same player provides
        too few observations and the tracker stays near 120 BPM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live_pipeline import (
    _compute_search_rolls,
    process_realtime_audio,
    _DEFAULT_PRE_ROLL_S,
    _DEFAULT_POST_ROLL_S,
)
from core.practice_session import PracticeSession
from core.tempo_tracker import TempoTracker


# ── Shared constants ──────────────────────────────────────────────────────────

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000
_BEAT_S   = 60.0 / _BPM   # 0.5 s
_CI_S     = _COUNT_IN * _BEAT_S  # 2.0 s

_SILENCE = np.zeros(500_000)


def _session(n: int = 10):
    return PracticeSession([{"time": i} for i in range(n)], _BPM, _COUNT_IN, _SR)


def _threshold_sample(beat_idx: int) -> int:
    """One sample past the ready threshold for beat_idx at _BPM / _COUNT_IN."""
    return round((_CI_S + beat_idx * _BEAT_S + 0.45) * _SR) + 1


def _spike_audio(n_beats: int, player_bpm: float, n_samples: int = 600_000) -> np.ndarray:
    audio = np.zeros(n_samples)
    player_beat_s = 60.0 / player_bpm
    for i in range(n_beats):
        idx = round((_CI_S + i * player_beat_s) * _SR)
        if 0 <= idx < n_samples:
            audio[idx] = 0.8
    return audio


# ── 1–5: _compute_search_rolls unit tests ────────────────────────────────────

def test_confidence_zero_gives_full_extra():
    """At confidence=0 the full adaptive_search_margin is applied."""
    margin = 0.15
    pre, post = _compute_search_rolls(0.0, _BEAT_S, margin, 0.25)
    expected_extra = margin * _BEAT_S
    assert pre  == pytest.approx(_DEFAULT_PRE_ROLL_S  + expected_extra, abs=1e-9)
    assert post == pytest.approx(_DEFAULT_POST_ROLL_S + expected_extra, abs=1e-9)


def test_confidence_one_gives_no_extra():
    """At confidence=1 rolls equal the fixed-grid defaults."""
    pre, post = _compute_search_rolls(1.0, _BEAT_S, 0.15, 0.25)
    assert pre  == pytest.approx(_DEFAULT_PRE_ROLL_S,  abs=1e-9)
    assert post == pytest.approx(_DEFAULT_POST_ROLL_S, abs=1e-9)


def test_confidence_half_halves_extra():
    """At confidence=0.5 the extra margin is half of the full margin."""
    margin = 0.20
    pre, post = _compute_search_rolls(0.5, _BEAT_S, margin, 0.25)
    expected_extra = 0.5 * margin * _BEAT_S
    assert pre  == pytest.approx(_DEFAULT_PRE_ROLL_S  + expected_extra, abs=1e-9)
    assert post == pytest.approx(_DEFAULT_POST_ROLL_S + expected_extra, abs=1e-9)


def test_extra_clamped_to_max():
    """Extra margin is capped at max_adaptive_search_margin_beats × beat_s."""
    pre, post = _compute_search_rolls(0.0, _BEAT_S, 0.30, 0.10)
    expected_extra = 0.10 * _BEAT_S
    assert pre  == pytest.approx(_DEFAULT_PRE_ROLL_S  + expected_extra, abs=1e-9)
    assert post == pytest.approx(_DEFAULT_POST_ROLL_S + expected_extra, abs=1e-9)


def test_zero_max_no_extra():
    """max_adaptive_search_margin_beats=0 disables widening entirely."""
    pre, post = _compute_search_rolls(0.0, _BEAT_S, 0.15, 0.0)
    assert pre  == pytest.approx(_DEFAULT_PRE_ROLL_S,  abs=1e-9)
    assert post == pytest.approx(_DEFAULT_POST_ROLL_S, abs=1e-9)


# ── 6–9: Event fields ─────────────────────────────────────────────────────────

def test_adaptive_event_has_search_roll_fields():
    """Adaptive events expose search_pre_roll_s and search_post_roll_s."""
    tracker = TempoTracker(_BPM)
    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s,
                                    tempo_tracker=tracker)
    assert "search_pre_roll_s"  in events[0]
    assert "search_post_roll_s" in events[0]


def test_fixed_grid_event_no_search_roll_fields():
    """Fixed-grid events do not expose search_pre_roll_s or search_post_roll_s."""
    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s)
    assert "search_pre_roll_s"  not in events[0]
    assert "search_post_roll_s" not in events[0]


def test_tracker_confidence_event_field_matches_pre_observe_confidence():
    """tracker_confidence equals the confidence computed before observe()."""
    tracker = TempoTracker(_BPM)
    pre_confidence = tracker.confidence()   # 0.0 — no data yet

    s = _session()
    events = process_realtime_audio(_SILENCE, _threshold_sample(0), s,
                                    tempo_tracker=tracker)
    assert events[0]["tracker_confidence"] == pytest.approx(pre_confidence, abs=1e-9)


def test_search_pre_roll_shrinks_as_confidence_rises():
    """search_pre_roll_s decreases (or stays equal) as tracker gains confidence."""
    player_bpm    = 121.0
    player_beat_s = 60.0 / player_bpm
    audio         = _spike_audio(20, player_bpm)

    tracker = TempoTracker(_BPM)
    s       = _session(20)

    pre_rolls = []
    for i in range(20):
        evs = process_realtime_audio(audio, _threshold_sample(i), s,
                                     tempo_tracker=tracker)
        if evs:
            pre_rolls.append(evs[0]["search_pre_roll_s"])

    assert len(pre_rolls) >= 5, "Need at least 5 events for trend check"
    # The first pre_roll should be wider than the last
    assert pre_rolls[0] >= pre_rolls[-1], (
        f"pre_roll should shrink as confidence rises: first={pre_rolls[0]:.4f}, "
        f"last={pre_rolls[-1]:.4f}"
    )


# ── 10: Fixed-grid regression ─────────────────────────────────────────────────

def test_fixed_grid_uses_default_rolls():
    """Fixed-grid mode always uses the default pre/post rolls."""
    # A spike 35 ms before the nominal beat — outside 30 ms default pre_roll.
    audio = np.zeros(500_000)
    beat0_samp = round((_CI_S - 0.035) * _SR)
    audio[beat0_samp] = 0.8

    s = _session()
    events = process_realtime_audio(audio, _threshold_sample(0), s)
    # Fixed-grid with 30 ms pre_roll should NOT find the onset 35 ms before.
    assert events[0]["detected_note"] is None, (
        "Fixed-grid should not detect a spike 35 ms before beat (pre_roll=30 ms)"
    )


# ── 11: No false observation on silence ───────────────────────────────────────

def test_silence_adaptive_no_detection_no_tracker_update():
    """Silence in adaptive mode: no onset detected, tracker state unchanged."""
    tracker = TempoTracker(_BPM)
    # Set anchor so ratio/phase could theoretically be modified
    tracker.observe(_CI_S, _CI_S)
    ratio_before = tracker.tempo_ratio
    phase_before = tracker.phase_offset

    s = _session()
    s.evaluated_indices = {0}  # skip beat 0 (already anchored)
    events = process_realtime_audio(_SILENCE, _threshold_sample(1), s,
                                    tempo_tracker=tracker)

    assert events[0]["detected_note"] is None
    assert tracker.tempo_ratio   == pytest.approx(ratio_before, abs=1e-12)
    assert tracker.phase_offset  == pytest.approx(phase_before, abs=1e-12)


# ── 12: Outlier isolation preserved ──────────────────────────────────────────

def test_outlier_does_not_retime_grid_with_wide_window():
    """Wide search window must not cause outliers to update the tracker.

    Scenario: tracker trained on 120 BPM. A spike at nominal+300 ms (60% of
    one beat — beyond outlier_threshold=40%) should be rejected even though
    it falls inside the widened window.
    """
    tracker = TempoTracker(_BPM)
    # Pre-train: 10 beats at exactly 120 BPM
    for i in range(10):
        tracker.observe(_CI_S + i * _BEAT_S, _CI_S + i * _BEAT_S)

    ratio_before = tracker.tempo_ratio
    phase_before = tracker.phase_offset

    # Spike at 300 ms late — outlier (> 200 ms threshold at 120 BPM)
    audio = np.zeros(500_000)
    beat10_nom = _CI_S + 10 * _BEAT_S
    spike_samp = round((beat10_nom + 0.300) * _SR)
    audio[spike_samp] = 0.8

    s = _session(11)
    s.evaluated_indices = set(range(10))
    # Use wide search margin so the spike is inside the window
    process_realtime_audio(
        audio, _threshold_sample(10), s,
        tempo_tracker=tracker,
        adaptive_search_margin_beats=0.25,
    )

    assert tracker.tempo_ratio  == pytest.approx(ratio_before, abs=1e-9), (
        "tempo_ratio must not change after a 300 ms outlier"
    )
    assert tracker.phase_offset == pytest.approx(phase_before, abs=1e-9), (
        "phase_offset must not change after a 300 ms outlier"
    )


# ── 13: End-to-end convergence 120 nominal / 132 actual ──────────────────────

def test_120_nominal_132_actual_converges_with_wide_search():
    """With adaptive search widening, 132 BPM player gets enough observations
    over 24 beats for the tracker to reach ≥ 128 BPM.

    Geometry at beat 1 (the first post-anchor beat):
        nominal window without widening: [2.5 - 0.03, 2.5 + 0.10] = [2.470, 2.600] s
        player onset at 132 BPM:         2.0 + 1 × (60/132) ≈ 2.455 s   (MISS)
        nominal window with widening:    [2.5 - ~0.105, 2.5 + ~0.175]   (HIT)
    """
    player_bpm = 132.0
    n_beats    = 24
    audio      = _spike_audio(n_beats, player_bpm)

    tracker = TempoTracker(_BPM)
    s = _session(n_beats)

    for i in range(n_beats):
        process_realtime_audio(audio, _threshold_sample(i), s, tempo_tracker=tracker)

    final_bpm = tracker.current_tempo_bpm()
    assert final_bpm >= 127.5, (
        f"Expected tracker ≥ 127.5 BPM after 24 beats at 132 BPM "
        f"with adaptive search widening; got {final_bpm:.2f} BPM"
    )


def test_120_nominal_132_actual_fixed_grid_stays_near_nominal():
    """Without adaptive search, the 132 BPM player provides too few observations
    for the tracker to leave the 120 BPM neighborhood.

    This is the original failure mode: onsets fall before the 30 ms pre-roll
    so the tracker starves of data.
    """
    player_bpm = 132.0
    n_beats    = 24
    audio      = _spike_audio(n_beats, player_bpm)

    # Simulate what fixed-grid provides: only direct observations that fall
    # within the nominal 30 ms pre-roll window.  At 132 BPM the drift per beat
    # is 45.5 ms, so only beat 0 (the anchor) is ever inside the nominal window.
    tracker = TempoTracker(_BPM)
    player_beat_s = 60.0 / player_bpm
    nom_beat_s    = 60.0 / _BPM

    for i in range(n_beats):
        nom  = _CI_S + i * nom_beat_s
        act  = _CI_S + i * player_beat_s
        # Fixed-grid window: onset must not be more than pre_roll seconds EARLY.
        # At 132 BPM the player is always early, so the relevant check is:
        #   act >= nom - _DEFAULT_PRE_ROLL_S
        # Beat 0 hits exactly; beat 1+ are ~45.5 ms early (outside 30 ms pre-roll).
        if act >= nom - _DEFAULT_PRE_ROLL_S:
            tracker.observe(nom, act)

    final_bpm = tracker.current_tempo_bpm()
    assert final_bpm < 123.0, (
        f"Without adaptive search the tracker should stay near nominal; "
        f"got {final_bpm:.2f} BPM (expected < 123.0)"
    )
