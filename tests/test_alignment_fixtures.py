"""Tests for alignment fixture files and the new helpers added to core/alignment.py.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Fixture loading
  1.  basic_four_beats_track.json loads without error.
  2.  eighth_notes_track.json loads without error.

Fixture BPM values
  3.  basic_four_beats_track: estimated BPM == 60.
  4.  eighth_notes_track: estimated BPM == 120.

Fixture fields
  5.  basic_four_beats_track: confirmed_by_user=True, confidence="fixture".
  6.  eighth_notes_track: confirmed_by_user=True, confidence="fixture".

beat_index_to_audio_time — integer indices
  7.  beat_index 0 returns first_beat_time_sec.
  8.  beat_index beat_count-1 returns last_beat_time_sec.
  9.  beat_index 1 returns first + 1 × period.

beat_index_to_audio_time — fractional indices
  10. beat_index 0.5 returns first + 0.5 × period.
  11. beat_index 1.5 returns first + 1.5 × period.

beat_index_to_audio_time — error cases
  12. Negative integer beat_index raises ValueError.
  13. Negative fractional beat_index raises ValueError.

exercise_target_audio_times — raw target dicts
  14. Raw target dicts produce correct audio times.
  15. Times are in target order.

exercise_target_audio_times — Exercise object
  16. Exercise object produces same times as equivalent raw dicts.
  17. Works with basic_four_beats.json exercise and basic_four_beats_track alignment.

alignment_to_exercise_clock
  18. Returns dict with "bpm" equal to estimated_bpm(alignment).
  19. Returns dict with "count_in_beats" == 0.
  20. Works with both fixture alignments.

Save / reload round-trip (fixture-style)
  21. A fixture-style alignment round-trips through save/load unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.alignment import (
    BeatAlignment,
    alignment_to_exercise_clock,
    beat_index_to_audio_time,
    beat_period_sec,
    estimated_bpm,
    exercise_target_audio_times,
    load_alignment_file,
    save_alignment_file,
)
from core.exercise import load_exercise_file, simple_timing_exercise

_ALIGNMENTS_DIR = Path(__file__).resolve().parents[1] / "alignments"
_EXERCISES_DIR  = Path(__file__).resolve().parents[1] / "exercises"

_BASIC_ALIGN   = _ALIGNMENTS_DIR / "basic_four_beats_track.json"
_EIGHTH_ALIGN  = _ALIGNMENTS_DIR / "eighth_notes_track.json"
_BASIC_EXERCISE = _EXERCISES_DIR / "basic_four_beats.json"


# ── 1–2: Fixture loading ──────────────────────────────────────────────────────

def test_basic_four_beats_track_loads():
    align = load_alignment_file(_BASIC_ALIGN)
    assert align.audio_file == "tracks/basic_four_beats.wav"


def test_eighth_notes_track_loads():
    align = load_alignment_file(_EIGHTH_ALIGN)
    assert align.audio_file == "tracks/eighth_notes_120.wav"


# ── 3–4: Fixture BPM values ───────────────────────────────────────────────────

def test_basic_four_beats_bpm():
    # first=2.0, last=5.0, count=4 → period=1.0 s → 60 BPM
    align = load_alignment_file(_BASIC_ALIGN)
    assert estimated_bpm(align) == pytest.approx(60.0)


def test_eighth_notes_bpm():
    # first=1.0, last=4.5, count=8 → period=0.5 s → 120 BPM
    align = load_alignment_file(_EIGHTH_ALIGN)
    assert estimated_bpm(align) == pytest.approx(120.0)


# ── 5–6: Fixture fields ───────────────────────────────────────────────────────

def test_basic_four_beats_fields():
    align = load_alignment_file(_BASIC_ALIGN)
    assert align.confirmed_by_user is True
    assert align.confidence == "fixture"


def test_eighth_notes_fields():
    align = load_alignment_file(_EIGHTH_ALIGN)
    assert align.confirmed_by_user is True
    assert align.confidence == "fixture"


# ── 7–9: beat_index_to_audio_time — integer indices ──────────────────────────

def test_beat_index_zero_returns_first():
    align = load_alignment_file(_BASIC_ALIGN)
    assert beat_index_to_audio_time(align, 0) == pytest.approx(align.first_beat_time_sec)


def test_beat_index_last_returns_last():
    align = load_alignment_file(_BASIC_ALIGN)
    assert beat_index_to_audio_time(align, align.beat_count - 1) == pytest.approx(
        align.last_beat_time_sec
    )


def test_beat_index_one():
    align = load_alignment_file(_BASIC_ALIGN)
    # period = 1.0 s; beat 1 → 2.0 + 1 × 1.0 = 3.0
    assert beat_index_to_audio_time(align, 1) == pytest.approx(3.0)


# ── 10–11: beat_index_to_audio_time — fractional indices ─────────────────────

def test_beat_index_half():
    align = load_alignment_file(_BASIC_ALIGN)
    # period = 1.0 s; beat 0.5 → 2.0 + 0.5 × 1.0 = 2.5
    assert beat_index_to_audio_time(align, 0.5) == pytest.approx(2.5)


def test_beat_index_one_and_half():
    align = load_alignment_file(_EIGHTH_ALIGN)
    # period = 0.5 s; beat 1.5 → 1.0 + 1.5 × 0.5 = 1.75
    assert beat_index_to_audio_time(align, 1.5) == pytest.approx(1.75)


# ── 12–13: beat_index_to_audio_time — error cases ────────────────────────────

def test_negative_integer_index_raises():
    align = load_alignment_file(_BASIC_ALIGN)
    with pytest.raises(ValueError, match="beat_index"):
        beat_index_to_audio_time(align, -1)


def test_negative_fractional_index_raises():
    align = load_alignment_file(_BASIC_ALIGN)
    with pytest.raises(ValueError, match="beat_index"):
        beat_index_to_audio_time(align, -0.5)


# ── 14–15: exercise_target_audio_times — raw dicts ───────────────────────────

def test_exercise_target_audio_times_raw_dicts():
    align = load_alignment_file(_BASIC_ALIGN)
    # period=1.0 s, first=2.0; targets at beats 0,1,2,3 → 2.0, 3.0, 4.0, 5.0
    targets = [{"time": 0}, {"time": 1}, {"time": 2}, {"time": 3}]
    times = exercise_target_audio_times(targets, align)
    assert times == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_exercise_target_audio_times_preserves_order():
    align = load_alignment_file(_EIGHTH_ALIGN)
    # period=0.5 s, first=1.0; targets at 0, 0.5, 1.0 → 1.0, 1.25, 1.5
    targets = [{"time": 0}, {"time": 0.5}, {"time": 1.0}]
    times = exercise_target_audio_times(targets, align)
    assert times == pytest.approx([1.0, 1.25, 1.5])


# ── 16–17: exercise_target_audio_times — Exercise object ─────────────────────

def test_exercise_target_audio_times_exercise_matches_raw():
    align = load_alignment_file(_BASIC_ALIGN)
    raw   = [{"time": 0}, {"time": 1}, {"time": 2}, {"time": 3}]
    ex    = simple_timing_exercise("test", 120.0, 0, [0.0, 1.0, 2.0, 3.0])
    assert exercise_target_audio_times(raw, align) == pytest.approx(
        exercise_target_audio_times(ex, align)
    )


def test_exercise_target_audio_times_with_loaded_exercise():
    align = load_alignment_file(_BASIC_ALIGN)
    ex    = load_exercise_file(_BASIC_EXERCISE)
    # basic_four_beats.json: targets at beats 0,1,2,3; alignment period=1.0 s, first=2.0
    times = exercise_target_audio_times(ex, align)
    assert times == pytest.approx([2.0, 3.0, 4.0, 5.0])


# ── 18–20: alignment_to_exercise_clock ───────────────────────────────────────

def test_exercise_clock_bpm():
    align = load_alignment_file(_BASIC_ALIGN)
    clock = alignment_to_exercise_clock(align)
    assert clock["bpm"] == pytest.approx(estimated_bpm(align))


def test_exercise_clock_count_in_zero():
    align = load_alignment_file(_BASIC_ALIGN)
    clock = alignment_to_exercise_clock(align)
    assert clock["count_in_beats"] == 0


def test_exercise_clock_both_fixtures():
    for path in (_BASIC_ALIGN, _EIGHTH_ALIGN):
        align = load_alignment_file(path)
        clock = alignment_to_exercise_clock(align)
        assert "bpm" in clock
        assert clock["count_in_beats"] == 0


# ── 21: Save / reload round-trip ─────────────────────────────────────────────

def test_fixture_style_save_reload(tmp_path):
    align = load_alignment_file(_BASIC_ALIGN)
    dest  = tmp_path / "copy.json"
    save_alignment_file(align, dest)
    align2 = load_alignment_file(dest)
    assert align == align2
