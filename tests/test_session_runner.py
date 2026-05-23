"""Tests for core/session_runner.py — pure session executor.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Bundle validation
  0.  Invalid bundle (metronome with no exercise) raises ValueError before any
      processing occurs.
  0b. Invalid bundle (metronome with alignment present) raises ValueError.
  0c. Valid bundles are not rejected.

Metronome mode
  1.  Happy path: all onsets within window → all target_hit events.
  2.  Two targets missed → two target_miss events with correct target_index.
  3.  One extra onset (outside all windows) → one extra_onset event.
  4.  Empty onset stream → all targets missed, no hits, no extras.

Recording-aligned mode
  5.  Happy path: onsets within alignment-derived windows → all target_hit.

play_to_align mode
  6.  No targets → all onsets become extra_onset events.
  7.  No targets and no onsets → empty event list.

Metrics
  8.  targets_total, targets_hit, targets_missed, extra_onsets are correct.

Event ordering
  9.  Events are sorted by time_sec regardless of hit/miss/extra mix.

Session log properties
  10. validate_session_log(result) raises no exception.
  11. started_at and ended_at are passed through correctly.
  12. exercise_path is copied from the practice mode.
  13. alignment_path is copied from the practice mode.
  14. target_hit value field holds timing error (positive = late).
  15. target_hit value field holds timing error (negative = early).
  16. target_miss value is None.
  17. extra_onset target_index is None.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.alignment import BeatAlignment
from core.exercise import Exercise, Target
from core.practice_mode import PracticeMode
from core.session_bundle import SessionBundle
from core.session_log import validate_session_log
from core.session_runner import run_session_bundle


# ── Shared helpers ────────────────────────────────────────────────────────────

# Exercise: bpm=120, count_in=2, targets at beats [0, 1, 2, 3]
# beat_s = 0.5 s  |  count_in_s = 1.0 s
# target audio times → [1.0, 1.5, 2.0, 2.5]  |  window = 30/120 = 0.25 s
def _metro_exercise() -> Exercise:
    return Exercise(
        schema_version = 1,
        name           = "Four Beats",
        bpm            = 120.0,
        count_in_beats = 2,
        targets        = [Target(time=float(i)) for i in range(4)],
    )


def _metro_mode() -> PracticeMode:
    return PracticeMode(
        schema_version = 1,
        mode           = "metronome_exercise",
        exercise_path  = "exercises/four_beats.json",
    )


def _metro_bundle() -> SessionBundle:
    return SessionBundle(
        practice_mode = _metro_mode(),
        exercise      = _metro_exercise(),
    )


# Alignment: first=2.0, last=5.0, beat_count=4 → period=1.0 s, bpm=60
# target audio times (beats 0–3) → [2.0, 3.0, 4.0, 5.0]  |  window = 0.5 s
def _alignment() -> BeatAlignment:
    return BeatAlignment(
        schema_version      = 1,
        audio_file          = "tracks/test.wav",
        alignment_method    = "manual_tap",
        first_beat_time_sec = 2.0,
        last_beat_time_sec  = 5.0,
        beat_count          = 4,
        confidence          = "high",
    )


def _aligned_mode() -> PracticeMode:
    return PracticeMode(
        schema_version = 1,
        mode           = "recording_aligned_exercise",
        exercise_path  = "exercises/four_beats.json",
        alignment_path = "alignments/four_beats_track.json",
    )


def _aligned_bundle() -> SessionBundle:
    return SessionBundle(
        practice_mode = _aligned_mode(),
        exercise      = _metro_exercise(),
        alignment     = _alignment(),
    )


def _play_bundle() -> SessionBundle:
    return SessionBundle(
        practice_mode = PracticeMode(
            schema_version = 1,
            mode           = "play_to_align",
            audio_file     = "tracks/test.wav",
        ),
    )


_STARTED = "2026-05-23T10:00:00"
_ENDED   = "2026-05-23T10:05:00"


# ── 1: metronome happy path ───────────────────────────────────────────────────

def test_metronome_happy_path_all_hits():
    # Target times: [1.0, 1.5, 2.0, 2.5]  window=0.25 s
    onsets = [1.01, 1.49, 2.02, 2.48]
    log = run_session_bundle(_metro_bundle(), onsets, started_at=_STARTED)

    hits = [e for e in log.events if e.event_type == "target_hit"]
    assert len(hits)   == 4
    assert len(log.events) == 4
    assert all(e.target_index is not None for e in hits)


# ── 2: misses ────────────────────────────────────────────────────────────────

def test_metronome_two_targets_missed():
    # Only provide onsets for targets 0 and 2; targets 1 and 3 are missed.
    # target 0 → 1.0 s, target 2 → 2.0 s
    onsets = [1.01, 2.01]
    log = run_session_bundle(_metro_bundle(), onsets, started_at=_STARTED)

    misses = [e for e in log.events if e.event_type == "target_miss"]
    assert len(misses) == 2
    assert {e.target_index for e in misses} == {1, 3}


def test_miss_event_time_is_nominal_target_time():
    # target 1 missed → event time should be the nominal target time (1.5 s)
    log = run_session_bundle(_metro_bundle(), [1.01, 2.01], started_at=_STARTED)
    miss_1 = next(e for e in log.events if e.event_type == "target_miss" and e.target_index == 1)
    assert miss_1.time_sec == pytest.approx(1.5)


# ── 3: extra onsets ───────────────────────────────────────────────────────────

def test_extra_onset_outside_all_windows():
    # onset at 0.5 s is 0.5 s from target 0 (1.0 s), outside window (0.25 s)
    onsets = [0.5, 1.01, 1.51, 2.01, 2.51]
    log = run_session_bundle(_metro_bundle(), onsets, started_at=_STARTED)

    extras = [e for e in log.events if e.event_type == "extra_onset"]
    hits   = [e for e in log.events if e.event_type == "target_hit"]
    assert len(extras) == 1
    assert extras[0].time_sec == pytest.approx(0.5)
    assert len(hits) == 4


# ── 4: empty onset stream ────────────────────────────────────────────────────

def test_empty_onset_stream_all_missed():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)

    misses = [e for e in log.events if e.event_type == "target_miss"]
    assert len(misses)     == 4
    assert len(log.events) == 4
    assert not any(e.event_type == "target_hit"    for e in log.events)
    assert not any(e.event_type == "extra_onset"   for e in log.events)


# ── 5: recording_aligned happy path ──────────────────────────────────────────

def test_recording_aligned_happy_path_all_hits():
    # Target times: [2.0, 3.0, 4.0, 5.0]  window=0.5 s (bpm=60)
    onsets = [2.05, 2.96, 4.03, 4.95]
    log = run_session_bundle(_aligned_bundle(), onsets, started_at=_STARTED)

    hits = [e for e in log.events if e.event_type == "target_hit"]
    assert len(hits)       == 4
    assert len(log.events) == 4


def test_recording_aligned_target_indices_correct():
    onsets = [2.05, 2.96, 4.03, 4.95]
    log = run_session_bundle(_aligned_bundle(), onsets, started_at=_STARTED)

    hit_indices = sorted(e.target_index for e in log.events if e.event_type == "target_hit")
    assert hit_indices == [0, 1, 2, 3]


# ── 6–7: play_to_align ───────────────────────────────────────────────────────

def test_play_to_align_all_onsets_are_extras():
    onsets = [0.5, 1.0, 1.5]
    log = run_session_bundle(_play_bundle(), onsets, started_at=_STARTED)

    extras = [e for e in log.events if e.event_type == "extra_onset"]
    assert len(extras)     == 3
    assert len(log.events) == 3
    assert not any(e.event_type in ("target_hit", "target_miss") for e in log.events)


def test_play_to_align_no_onsets_empty_log():
    log = run_session_bundle(_play_bundle(), [], started_at=_STARTED)
    assert log.events == []


# ── 8: metrics ───────────────────────────────────────────────────────────────

def test_metrics_correctness():
    # Targets at [1.0, 1.5, 2.0]  (beats 0,1,2; bpm=120, count_in=2)
    # window = 0.25 s
    ex = Exercise(
        schema_version = 1,
        name           = "Three Beats",
        bpm            = 120.0,
        count_in_beats = 2,
        targets        = [Target(time=float(i)) for i in range(3)],
    )
    bundle = SessionBundle(
        practice_mode = PracticeMode(
            schema_version = 1,
            mode           = "metronome_exercise",
            exercise_path  = "exercises/three.json",
        ),
        exercise = ex,
    )

    # onset 1.02 → hits target 0 (1.0 s)
    # onset 1.80 → hits target 2 (2.0 s)  [|1.8-2.0|=0.2 < 0.25]
    # onset 2.01 → target 2 already taken; |2.01-1.5|=0.51 > 0.25 → extra
    # target 1 (1.5 s) has no close onset → miss
    onsets = [1.02, 1.80, 2.01]
    log = run_session_bundle(bundle, onsets, started_at=_STARTED)

    assert log.metrics["targets_total"]  == 3
    assert log.metrics["targets_hit"]    == 2
    assert log.metrics["targets_missed"] == 1
    assert log.metrics["extra_onsets"]   == 1


# ── 9: event ordering ────────────────────────────────────────────────────────

def test_events_sorted_by_time_sec():
    # Targets at [1.0, 2.0]  window=0.25 s
    # onset 2.1 → hits target 1 (delta=0.1)
    # onset 1.7 → |1.7-1.0|=0.7 and |1.7-2.0|=0.3 — both > 0.25 → extra
    # target 0 (1.0 s) has no close onset → miss
    ex = Exercise(
        schema_version = 1,
        name           = "Two Beats",
        bpm            = 120.0,
        count_in_beats = 2,
        targets        = [Target(time=0.0), Target(time=2.0)],
    )
    bundle = SessionBundle(
        practice_mode = PracticeMode(
            schema_version = 1,
            mode           = "metronome_exercise",
            exercise_path  = "exercises/two.json",
        ),
        exercise = ex,
    )

    log = run_session_bundle(bundle, [2.1, 1.7], started_at=_STARTED)
    times = [e.time_sec for e in log.events]
    assert times == sorted(times)


def test_events_sorted_mixed_types_order():
    # Verify specific ordering: miss(1.0) < extra(1.7) < hit(2.1)
    ex = Exercise(
        schema_version = 1,
        name           = "Two Beats",
        bpm            = 120.0,
        count_in_beats = 2,
        targets        = [Target(time=0.0), Target(time=2.0)],
    )
    bundle = SessionBundle(
        practice_mode = PracticeMode(
            schema_version = 1,
            mode           = "metronome_exercise",
            exercise_path  = "exercises/two.json",
        ),
        exercise = ex,
    )

    log = run_session_bundle(bundle, [2.1, 1.7], started_at=_STARTED)
    types = [e.event_type for e in log.events]
    assert types == ["target_miss", "extra_onset", "target_hit"]


# ── 10–13: session log properties ────────────────────────────────────────────

def test_validate_session_log_passes():
    log = run_session_bundle(_metro_bundle(), [1.0, 1.5, 2.0, 2.5], started_at=_STARTED)
    validate_session_log(log)  # must not raise


def test_started_at_passed_through():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)
    assert log.started_at == _STARTED


def test_ended_at_passed_through():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED, ended_at=_ENDED)
    assert log.ended_at == _ENDED


def test_ended_at_none_by_default():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)
    assert log.ended_at is None


def test_exercise_path_copied_from_practice_mode():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)
    assert log.exercise_path == "exercises/four_beats.json"


def test_alignment_path_copied_from_practice_mode():
    log = run_session_bundle(_aligned_bundle(), [], started_at=_STARTED)
    assert log.alignment_path == "alignments/four_beats_track.json"


def test_alignment_path_none_for_metronome():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)
    assert log.alignment_path is None


# ── 14–17: event field correctness ───────────────────────────────────────────

def test_hit_value_is_positive_timing_error_when_late():
    # Target 0 at 1.0 s; onset at 1.05 s → late by 0.05 s
    log = run_session_bundle(_metro_bundle(), [1.05, 1.5, 2.0, 2.5], started_at=_STARTED)
    hit_0 = next(e for e in log.events if e.event_type == "target_hit" and e.target_index == 0)
    assert hit_0.value == pytest.approx(0.05)


def test_hit_value_is_negative_timing_error_when_early():
    # Target 0 at 1.0 s; onset at 0.95 s → early by 0.05 s
    log = run_session_bundle(_metro_bundle(), [0.95, 1.5, 2.0, 2.5], started_at=_STARTED)
    hit_0 = next(e for e in log.events if e.event_type == "target_hit" and e.target_index == 0)
    assert hit_0.value == pytest.approx(-0.05)


def test_miss_value_is_none():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)
    miss = log.events[0]
    assert miss.event_type == "target_miss"
    assert miss.value is None


def test_extra_onset_target_index_is_none():
    log = run_session_bundle(_play_bundle(), [0.5], started_at=_STARTED)
    extra = log.events[0]
    assert extra.event_type    == "extra_onset"
    assert extra.target_index  is None


# ── 0: bundle validation at entry ─────────────────────────────────────────────

def test_invalid_bundle_no_exercise_raises_before_processing():
    # metronome_exercise with no exercise — validate_session_bundle should fire
    # immediately and raise ValueError before any matching or event creation.
    bundle = SessionBundle(practice_mode=_metro_mode())  # exercise=None
    with pytest.raises(ValueError, match="exercise"):
        run_session_bundle(bundle, [1.0, 1.5, 2.0, 2.5], started_at=_STARTED)


def test_invalid_bundle_metronome_with_alignment_raises():
    # metronome_exercise must not have an alignment.
    bundle = SessionBundle(
        practice_mode = _metro_mode(),
        exercise      = _metro_exercise(),
        alignment     = _alignment(),
    )
    with pytest.raises(ValueError, match="alignment"):
        run_session_bundle(bundle, [], started_at=_STARTED)


def test_invalid_bundle_aligned_no_alignment_raises():
    # recording_aligned_exercise requires both exercise and alignment.
    bundle = SessionBundle(
        practice_mode = _aligned_mode(),
        exercise      = _metro_exercise(),
        # alignment intentionally absent
    )
    with pytest.raises(ValueError, match="alignment"):
        run_session_bundle(bundle, [], started_at=_STARTED)


def test_valid_metronome_bundle_not_rejected():
    log = run_session_bundle(_metro_bundle(), [], started_at=_STARTED)
    assert log.schema_version == 1


def test_valid_aligned_bundle_not_rejected():
    log = run_session_bundle(_aligned_bundle(), [], started_at=_STARTED)
    assert log.schema_version == 1


def test_valid_play_to_align_bundle_not_rejected():
    log = run_session_bundle(_play_bundle(), [], started_at=_STARTED)
    assert log.schema_version == 1
