"""Sequence-level tests for SessionEngine driven by simulated onset streams.

All tests are deterministic — no audio hardware, no threads, no sleeps.
Random jitter scenarios use fixed seeds.

Scenarios
---------
  1.  Perfectly on-time playing.
  2.  Consistently late playing.
  3.  Gradual tempo acceleration — all targets matched.
  4.  Gradual tempo acceleration — tracker detects tempo increase.
  5.  One missed note in the middle.
  6.  Multiple missed notes.
  7.  Extra unmatched onset between sparse targets.
  8.  Jittered human timing (±30 ms, fixed seed).
  9.  Sparse targets with long rests — no spurious misses.
  10. Dense 8th-note targets — custom narrow match window.
  11. Consistently late — tracker phase absorbed by anchor (zero residuals).
  12. No duplicate evaluations across a full sequence.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# tests/ dir — so we can import helpers.onset_stream
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.session_engine import SessionEngine
from core.tempo_tracker import TempoTracker
from helpers.onset_stream import simulate_onset_stream

# ── Shared constants ──────────────────────────────────────────────────────────

NOM_BPM    = 120.0
COUNT_IN   = 2
BEAT_S     = 60.0 / NOM_BPM      # 0.5 s
COUNT_IN_S = COUNT_IN * BEAT_S    # 1.0 s
HALF_BEAT  = BEAT_S * 0.5         # 0.25 s  (default match_window_s)

# Default outlier limit inside TempoTracker: 0.40 × beat_s = 200 ms.
TRACKER_OUTLIER_S = 0.40 * BEAT_S


# ── Shared helpers ────────────────────────────────────────────────────────────

def nom(i: int) -> float:
    """Nominal onset time (seconds) for target at beat index *i*."""
    return COUNT_IN_S + i * BEAT_S


def targets_n(n: int, step_beats: float = 1.0) -> list[dict]:
    """Return *n* targets spaced *step_beats* apart."""
    return [{"time": i * step_beats} for i in range(n)]


def session_end(targets: list[dict], extra_beats: float = 4.0) -> float:
    """Simulation end time: last target's nominal time plus *extra_beats*."""
    last = max(t["time"] for t in targets) if targets else 0.0
    return COUNT_IN_S + last * BEAT_S + extra_beats * BEAT_S


def make_engine(targets: list[dict], **kwargs) -> SessionEngine:
    return SessionEngine(targets, NOM_BPM, COUNT_IN, **kwargs)


def accel_onsets(
    start_bpm: float,
    end_bpm: float,
    n: int,
    count_in_s: float = COUNT_IN_S,
) -> list[float]:
    """Onset times for a player whose IOI interpolates linearly from
    *start_bpm* to *end_bpm* across *n* beats.

    Beat 0 is at the count-in anchor.  Each subsequent IOI is computed
    from the BPM at that beat index; the BPM fraction advances only after
    the interval is used (matching ``scenario_gradual_change`` in the replay
    tool).
    """
    times: list[float] = []
    t = count_in_s
    for i in range(n):
        times.append(t)
        frac = i / max(1, n - 1)
        current_bpm = start_bpm + frac * (end_bpm - start_bpm)
        t += 60.0 / current_bpm
    return times


def hits(events: list[dict]) -> list[dict]:
    return [e for e in events if e["detected_note"] is not None]


def misses(events: list[dict]) -> list[dict]:
    return [e for e in events if e["detected_note"] is None]


def assert_no_duplicates(events: list[dict]) -> None:
    indices = [e["target_idx"] for e in events]
    assert len(indices) == len(set(indices)), (
        f"Duplicate target_idx in events: {indices}"
    )


# ── 1. Perfectly on-time playing ─────────────────────────────────────────────

def test_perfectly_on_time_all_hit():
    n = 8
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == n
    assert len(misses(events)) == 0


def test_perfectly_on_time_target_order():
    n = 8
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert [e["target_idx"] for e in hits(events)] == list(range(n))


def test_perfectly_on_time_zero_error():
    n = 8
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    for ev in hits(events):
        assert abs(ev["timing_error_s"]) < 1e-6


def test_perfectly_on_time_no_duplicates():
    n = 8
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert_no_duplicates(events)
    assert len(eng.evaluated_indices) == n


# ── 2. Consistently late playing ─────────────────────────────────────────────

def test_consistently_late_all_matched():
    """80 ms late is within the half-beat match window (250 ms)."""
    n = 8
    offset = 0.080
    tgts = targets_n(n)
    onsets = [nom(i) + offset for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == n
    assert len(misses(events)) == 0


def test_consistently_late_timing_errors():
    n = 8
    offset = 0.080
    tgts = targets_n(n)
    onsets = [nom(i) + offset for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    for ev in hits(events):
        assert ev["timing_error_s"] == pytest.approx(offset, abs=1e-9)
    assert all(ev["severity"] == "warn" for ev in hits(events))


def test_consistently_late_no_duplicates():
    n = 8
    offset = 0.080
    tgts = targets_n(n)
    onsets = [nom(i) + offset for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert_no_duplicates(events)


# ── 3 & 4. Gradual tempo acceleration ────────────────────────────────────────
#
# 120 → 126 BPM over 16 beats.
# Maximum cumulative drift ≈ 0.153 s < default match_window (0.25 s)
# and < default tracker outlier limit (0.20 s), so all observations
# are accepted.

def test_accel_all_targets_matched():
    n = 16
    tgts = targets_n(n)
    onsets = accel_onsets(120.0, 126.0, n)
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == n
    assert len(misses(events)) == 0


def test_accel_no_duplicates():
    n = 16
    tgts = targets_n(n)
    onsets = accel_onsets(120.0, 126.0, n)
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert_no_duplicates(events)


def test_accel_tracker_detects_tempo_increase():
    """Tracker should detect the player is running faster than nominal.

    After 16 beats of 120→126 BPM acceleration, tempo_ratio < 1.0 and
    current_tempo_bpm() > NOM_BPM.
    """
    n = 16
    tgts = targets_n(n)
    onsets = accel_onsets(120.0, 126.0, n)
    tracker = TempoTracker(NOM_BPM)
    eng = make_engine(tgts, tracker=tracker)

    simulate_onset_stream(eng, onsets, session_end(tgts))

    assert tracker.tempo_ratio < 1.0, (
        f"Expected tempo_ratio < 1.0 (faster player); got {tracker.tempo_ratio:.4f}"
    )
    assert tracker.current_tempo_bpm() > NOM_BPM, (
        f"Expected BPM > {NOM_BPM}; got {tracker.current_tempo_bpm():.2f}"
    )


def test_accel_errors_progressively_earlier():
    """Timing errors should become increasingly negative as acceleration
    accumulates (player drifts ahead of the nominal grid)."""
    n = 16
    tgts = targets_n(n)
    onsets = accel_onsets(120.0, 126.0, n)
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))
    errors = [ev["timing_error_s"] for ev in hits(events)]

    # Beat 0 has zero error (anchor); error should be monotonically non-increasing
    # from about beat 2 onwards.  Check that the last beat is earlier than the first.
    assert errors[-1] < errors[0], (
        f"Expected last error {errors[-1]:.4f} < first error {errors[0]:.4f}"
    )


# ── 5. One missed note in the middle ─────────────────────────────────────────

def test_one_missed_note_produces_miss_event():
    n = 8
    missed_idx = 3
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n) if i != missed_idx]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == n - 1
    assert len(misses(events)) == 1
    assert misses(events)[0]["target_idx"] == missed_idx
    assert misses(events)[0]["severity"] == "miss"


def test_one_missed_note_total_event_count():
    n = 8
    missed_idx = 3
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n) if i != missed_idx]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(events) == n
    assert_no_duplicates(events)


# ── 6. Multiple missed notes ──────────────────────────────────────────────────

def test_multiple_missed_notes():
    n = 8
    missed = {2, 5}
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n) if i not in missed]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    miss_events = misses(events)
    assert len(miss_events) == len(missed)
    assert {ev["target_idx"] for ev in miss_events} == missed
    assert len(hits(events)) == n - len(missed)
    assert_no_duplicates(events)


# ── 7. Extra unmatched onset between sparse targets ───────────────────────────
#
# Targets are 4 beats (2 s) apart.  An extra onset placed exactly at the
# midpoint between targets 0 and 1 is 1.0 s from each — far outside the
# default 0.25 s match window.

def test_extra_unmatched_onset_is_ignored():
    tgts = [{"time": 0}, {"time": 4}, {"time": 8}, {"time": 12}]
    nom_t = lambda i: COUNT_IN_S + tgts[i]["time"] * BEAT_S
    target_onsets = [nom_t(i) for i in range(4)]

    # Midpoint between target 0 (1.0 s) and target 1 (3.0 s)
    extra = (nom_t(0) + nom_t(1)) / 2.0   # 2.0 s — 1.0 s from each target

    end_s = session_end(tgts)
    eng = make_engine(tgts)
    events = simulate_onset_stream(eng, target_onsets + [extra], end_s)

    assert len(hits(events)) == 4
    assert len(misses(events)) == 0
    assert len(eng.evaluated_indices) == 4
    assert_no_duplicates(events)


def test_extra_unmatched_onset_does_not_consume_target():
    """The extra onset must not steal the match from a nearby target."""
    tgts = [{"time": 0}, {"time": 4}]
    nom_t = lambda i: COUNT_IN_S + tgts[i]["time"] * BEAT_S

    # Onset slightly after target 0's window end, clearly before target 1's window.
    extra = nom_t(0) + HALF_BEAT + 0.05   # 0.30 s after target 0

    eng = make_engine(tgts)
    events = simulate_onset_stream(
        eng,
        [nom_t(0), extra, nom_t(1)],
        session_end(tgts),
    )

    assert len(hits(events)) == 2
    assert hits(events)[0]["target_idx"] == 0
    assert hits(events)[1]["target_idx"] == 1


# ── 8. Jittered human timing ──────────────────────────────────────────────────

def test_jittered_timing_all_matched():
    """±30 ms jitter around each nominal time should match all 16 targets.

    Jitter (30 ms) is well inside the default match window (250 ms) and
    tracker outlier limit (200 ms), so every onset is accepted.
    """
    n = 16
    jitter_s = 0.030
    tgts = targets_n(n)
    rng = random.Random(42)
    onsets = [nom(i) + rng.uniform(-jitter_s, jitter_s) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == n
    assert len(misses(events)) == 0


def test_jittered_timing_errors_within_jitter_range():
    n = 16
    jitter_s = 0.030
    tgts = targets_n(n)
    rng = random.Random(42)
    onsets = [nom(i) + rng.uniform(-jitter_s, jitter_s) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    for ev in hits(events):
        assert abs(ev["timing_error_s"]) <= jitter_s + 1e-9


def test_jittered_timing_no_duplicates():
    n = 16
    jitter_s = 0.030
    tgts = targets_n(n)
    rng = random.Random(42)
    onsets = [nom(i) + rng.uniform(-jitter_s, jitter_s) for i in range(n)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert_no_duplicates(events)
    assert len(eng.evaluated_indices) == n


# ── 9. Sparse targets with long rests ────────────────────────────────────────
#
# Four targets spaced 4 beats (2 s) apart.  The 20 ms tick must not emit
# spurious miss events during the long silences between targets.

def test_sparse_targets_no_hits_missed():
    tgts = [{"time": 0}, {"time": 4}, {"time": 8}, {"time": 12}]
    nom_t = lambda i: COUNT_IN_S + tgts[i]["time"] * BEAT_S
    onsets = [nom_t(i) for i in range(4)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == 4
    assert len(misses(events)) == 0


def test_sparse_targets_correct_order():
    tgts = [{"time": 0}, {"time": 4}, {"time": 8}, {"time": 12}]
    nom_t = lambda i: COUNT_IN_S + tgts[i]["time"] * BEAT_S
    onsets = [nom_t(i) for i in range(4)]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert [ev["target_idx"] for ev in hits(events)] == [0, 1, 2, 3]


# ── 10. Dense 8th-note targets — narrow match window ─────────────────────────
#
# 8th notes are spaced BEAT_S / 2 = 0.25 s apart.  The default
# match_window_s (0.25 s) equals the inter-target spacing, so windows
# fully overlap at their midpoints.  A narrow custom window (0.10 s)
# prevents cross-target capture while still accepting ±20 ms jitter.

def test_dense_8th_notes_narrow_window_all_hit():
    n = 8
    tgts = [{"time": i * 0.5} for i in range(n)]   # half-beat increments
    nom_8 = lambda i: COUNT_IN_S + i * (BEAT_S / 2)
    rng = random.Random(99)
    jitter_s = 0.020
    onsets = [nom_8(i) + rng.uniform(-jitter_s, jitter_s) for i in range(n)]
    end_s = nom_8(n - 1) + 4 * BEAT_S
    eng = make_engine(tgts, match_window_s=0.10)

    events = simulate_onset_stream(eng, onsets, end_s)

    assert len(hits(events)) == n
    assert len(misses(events)) == 0


def test_dense_8th_notes_correct_target_assignment():
    n = 8
    tgts = [{"time": i * 0.5} for i in range(n)]
    nom_8 = lambda i: COUNT_IN_S + i * (BEAT_S / 2)
    rng = random.Random(99)
    jitter_s = 0.020
    onsets = [nom_8(i) + rng.uniform(-jitter_s, jitter_s) for i in range(n)]
    end_s = nom_8(n - 1) + 4 * BEAT_S
    eng = make_engine(tgts, match_window_s=0.10)

    events = simulate_onset_stream(eng, onsets, end_s)

    assert [ev["target_idx"] for ev in hits(events)] == list(range(n))
    assert_no_duplicates(events)


def test_dense_8th_notes_default_window_would_overlap():
    """Confirm that the default match_window equals inter-target spacing for 8th notes,
    demonstrating why a narrow window is needed for dense targets."""
    tgts = [{"time": i * 0.5} for i in range(2)]
    eng_default = make_engine(tgts)
    spacing = BEAT_S / 2   # 0.25 s

    # Default match_window_s == spacing: windows touch exactly at the midpoint.
    assert eng_default.match_window_s == pytest.approx(spacing, abs=1e-9)


# ── 11. Tracker phase offset absorbed by anchor ───────────────────────────────
#
# When a player is consistently late by a fixed offset, the anchor (first
# observation) absorbs that offset.  All subsequent predictions are exact,
# prediction errors stay at zero, and phase_offset does not accumulate.

def test_consistently_late_tracker_absorbs_phase():
    n = 8
    offset = 0.080
    tgts = targets_n(n)
    onsets = [nom(i) + offset for i in range(n)]
    tracker = TempoTracker(NOM_BPM)
    eng = make_engine(tgts, tracker=tracker)

    simulate_onset_stream(eng, onsets, session_end(tgts))

    # Anchor absorbs offset → residuals ≈ 0 → tempo_ratio stays near 1.0
    assert tracker.has_anchor
    assert tracker.tempo_ratio == pytest.approx(1.0, abs=0.02)
    # Phase offset should remain near zero (anchor absorbed the constant shift)
    assert abs(tracker.phase_offset) < 0.01


# ── 13. Tracker-adjusted matching — 120 → 132 BPM ────────────────────────────
#
# At 132 BPM the player drifts ahead of nominal by roughly 285 ms at beat 14
# and 330 ms at beat 15 — both beyond the default half-beat match window
# (250 ms).  Without a tracker the engine misses those beats.  With a
# tracker the match window follows the adjusted prediction and all 16 are hit.

def test_accel_132_with_tracker_no_spurious_misses():
    """Tracker-adjusted matching recovers beats that fall outside the nominal window."""
    n = 16
    tgts = targets_n(n)
    onsets = accel_onsets(120.0, 132.0, n)
    tracker = TempoTracker(NOM_BPM)
    eng = make_engine(tgts, tracker=tracker)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(hits(events)) == n,  f"Expected {n} hits; got {len(hits(events))}"
    assert len(misses(events)) == 0, f"Expected 0 misses; got {len(misses(events))}"
    assert_no_duplicates(events)


def test_accel_132_without_tracker_late_beats_missed():
    """Without a tracker the same stream exposes two nominal-matching failures.

    At 120→132 BPM the drift reaches ~286 ms at beat 14 — just outside the
    default 250 ms window.  Beat 14's onset goes unmatched.  Beat 15's onset
    then arrives 171 ms after target 14's nominal, which IS within the window,
    so it steals target 14's slot with a misleading +171 ms error.  Target 15
    gets no onset and is declared a miss.

    Net result: ≥1 miss and ≥1 misrouted hit (target 14 matched by the wrong
    onset).  With a tracker neither failure occurs.
    """
    n = 16
    tgts = targets_n(n)
    onsets = accel_onsets(120.0, 132.0, n)
    eng = make_engine(tgts)   # no tracker

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert len(misses(events)) >= 1, (
        f"Expected ≥1 miss without tracker; got {len(misses(events))}"
    )
    # Target 14 is matched by beat 15's onset → positive error despite the
    # player being ahead of the grid.  Confirm misrouting occurred.
    t14 = next((e for e in hits(events) if e["target_idx"] == 14), None)
    assert t14 is not None and t14["timing_error_s"] > 0, (
        "Target 14 should be misrouted with a positive timing error"
    )


def test_adjusted_matching_timing_error_relative_to_nominal():
    """timing_error_s must be onset_time_s - nominal, not onset_time_s - adjusted.

    With the player at 132 BPM the adjusted prediction is early relative to
    the nominal grid.  The reported timing_error_s should reflect the musical
    reality (how far from the score), not the tracker's internal reference.
    """
    n = 8
    actual_bpm = 130.0
    tgts = targets_n(n)
    tracker = TempoTracker(NOM_BPM)
    eng = make_engine(tgts, tracker=tracker)

    act_beat_s = 60.0 / actual_bpm
    onsets = [COUNT_IN_S + i * act_beat_s for i in range(n)]

    events = simulate_onset_stream(eng, onsets, session_end(tgts))
    hit_events = hits(events)

    for ev in hit_events:
        idx = ev["target_idx"]
        expected_nominal_error = onsets[idx] - (COUNT_IN_S + idx * BEAT_S)
        assert ev["timing_error_s"] == pytest.approx(expected_nominal_error, abs=1e-9), (
            f"beat {idx}: timing_error_s should be relative to nominal"
        )


def test_adjusted_matching_fallback_before_anchor():
    """Before the tracker anchor is set (first beat), matching uses nominal time.

    We verify this by comparing engine output with and without a fresh tracker
    on a single-beat session: both should produce identical timing_error_s.
    """
    tgts = targets_n(1)
    onset = nom(0) + 0.040   # 40 ms late

    eng_no_tracker = make_engine(tgts)
    eng_with_tracker = make_engine(tgts, tracker=TempoTracker(NOM_BPM))

    ev_no  = eng_no_tracker.on_onset(onset)
    ev_yes = eng_with_tracker.on_onset(onset)

    assert len(ev_no) == 1 and len(ev_yes) == 1
    assert ev_no[0]["timing_error_s"] == pytest.approx(
        ev_yes[0]["timing_error_s"], abs=1e-12
    )


# ── 12. No duplicate evaluations across a full mixed sequence ─────────────────

def test_no_duplicate_evaluations_mixed_sequence():
    """Hit some beats, miss others — every target_idx appears exactly once."""
    n = 12
    missed = {1, 4, 9}
    tgts = targets_n(n)
    onsets = [nom(i) for i in range(n) if i not in missed]
    eng = make_engine(tgts)

    events = simulate_onset_stream(eng, onsets, session_end(tgts))

    assert_no_duplicates(events)
    assert len(events) == n
    assert len(eng.evaluated_indices) == n
    assert {ev["target_idx"] for ev in misses(events)} == missed
