"""Tests for core/session_bundle.py — practice session orchestration layer.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
validate_session_bundle — mode requirements
  1.  Valid metronome bundle (exercise present, no alignment).
  2.  Valid recording_aligned bundle (exercise + alignment present).
  3.  Valid play_to_align bundle (no exercise, no alignment).
  4.  metronome_exercise with exercise=None raises ValueError.
  5.  metronome_exercise with alignment present raises ValueError.
  6.  recording_aligned_exercise with exercise=None raises ValueError.
  7.  recording_aligned_exercise with alignment=None raises ValueError.
  8.  play_to_align with exercise present raises ValueError.
  9.  play_to_align with alignment present raises ValueError.

validate_session_bundle — session_log
  10. Bundle with a valid SessionLog is accepted.

load_session_bundle — loading
  11. Loads metronome bundle from a practice mode file (relative exercise_path).
  12. Loads recording_aligned bundle (relative exercise_path + alignment_path).
  13. Loads play_to_align bundle (no exercise or alignment files needed).

load_session_bundle — path resolution
  14. Relative paths are resolved relative to the practice mode file's directory.

load_session_bundle — error cases
  15. Missing exercise file raises FileNotFoundError with informative message.
  16. Missing alignment file raises FileNotFoundError with informative message.

bundle_target_audio_times
  17. Metronome mode: times derived from exercise bpm and count-in.
  18. Recording_aligned mode: times from exercise_target_audio_times.
  19. play_to_align returns empty list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.alignment import BeatAlignment
from core.exercise import Exercise, Target
from core.practice_mode import PracticeMode
from core.session_bundle import (
    SessionBundle,
    bundle_target_audio_times,
    load_session_bundle,
    validate_session_bundle,
)
from core.session_log import SessionLog


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _exercise() -> Exercise:
    return Exercise(
        schema_version = 1,
        name           = "Four Beats",
        bpm            = 120.0,
        count_in_beats = 2,
        targets        = [Target(time=float(i)) for i in range(4)],
    )


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


def _metronome_mode() -> PracticeMode:
    return PracticeMode(
        schema_version = 1,
        mode           = "metronome_exercise",
        exercise_path  = "exercises/ex.json",
    )


def _aligned_mode() -> PracticeMode:
    return PracticeMode(
        schema_version = 1,
        mode           = "recording_aligned_exercise",
        exercise_path  = "exercises/ex.json",
        alignment_path = "alignments/al.json",
    )


def _play_to_align_mode() -> PracticeMode:
    return PracticeMode(
        schema_version = 1,
        mode           = "play_to_align",
        audio_file     = "tracks/test.wav",
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _exercise_json() -> dict:
    return {
        "schema_version": 1,
        "name":           "Four Beats",
        "bpm":            120.0,
        "count_in_beats": 2,
        "targets": [{"time": 0}, {"time": 1}, {"time": 2}, {"time": 3}],
    }


def _alignment_json() -> dict:
    return {
        "schema_version":     1,
        "audio_file":         "tracks/test.wav",
        "alignment_method":   "manual_tap",
        "first_beat_time_sec": 2.0,
        "last_beat_time_sec":  5.0,
        "beat_count":          4,
        "beats_per_bar":       4,
        "confirmed_by_user":   True,
        "confidence":          "high",
    }


# ── 1–3: validate_session_bundle — valid bundles ──────────────────────────────

def test_validate_valid_metronome_bundle():
    bundle = SessionBundle(
        practice_mode = _metronome_mode(),
        exercise      = _exercise(),
    )
    validate_session_bundle(bundle)  # must not raise


def test_validate_valid_recording_aligned_bundle():
    bundle = SessionBundle(
        practice_mode = _aligned_mode(),
        exercise      = _exercise(),
        alignment     = _alignment(),
    )
    validate_session_bundle(bundle)  # must not raise


def test_validate_valid_play_to_align_bundle():
    bundle = SessionBundle(practice_mode=_play_to_align_mode())
    validate_session_bundle(bundle)  # must not raise


# ── 4–5: metronome_exercise requirements ──────────────────────────────────────

def test_metronome_no_exercise_raises():
    bundle = SessionBundle(practice_mode=_metronome_mode())
    with pytest.raises(ValueError, match="exercise"):
        validate_session_bundle(bundle)


def test_metronome_with_alignment_raises():
    bundle = SessionBundle(
        practice_mode = _metronome_mode(),
        exercise      = _exercise(),
        alignment     = _alignment(),
    )
    with pytest.raises(ValueError, match="alignment"):
        validate_session_bundle(bundle)


# ── 6–7: recording_aligned_exercise requirements ──────────────────────────────

def test_aligned_no_exercise_raises():
    bundle = SessionBundle(
        practice_mode = _aligned_mode(),
        alignment     = _alignment(),
    )
    with pytest.raises(ValueError, match="exercise"):
        validate_session_bundle(bundle)


def test_aligned_no_alignment_raises():
    bundle = SessionBundle(
        practice_mode = _aligned_mode(),
        exercise      = _exercise(),
    )
    with pytest.raises(ValueError, match="alignment"):
        validate_session_bundle(bundle)


# ── 8–9: play_to_align requirements ──────────────────────────────────────────

def test_play_to_align_with_exercise_raises():
    bundle = SessionBundle(
        practice_mode = _play_to_align_mode(),
        exercise      = _exercise(),
    )
    with pytest.raises(ValueError, match="exercise"):
        validate_session_bundle(bundle)


def test_play_to_align_with_alignment_raises():
    bundle = SessionBundle(
        practice_mode = _play_to_align_mode(),
        alignment     = _alignment(),
    )
    with pytest.raises(ValueError, match="alignment"):
        validate_session_bundle(bundle)


# ── 10: session_log accepted ──────────────────────────────────────────────────

def test_bundle_with_session_log_accepted():
    log = SessionLog(schema_version=1, started_at="2026-05-23T10:00:00")
    bundle = SessionBundle(
        practice_mode = _metronome_mode(),
        exercise      = _exercise(),
        session_log   = log,
    )
    validate_session_bundle(bundle)  # must not raise


# ── 11–13: load_session_bundle ────────────────────────────────────────────────

def test_load_metronome_bundle(tmp_path):
    modes_dir = tmp_path / "modes"
    ex_dir    = tmp_path / "exercises"

    _write_json(ex_dir / "ex.json", _exercise_json())
    _write_json(modes_dir / "session.json", {
        "schema_version": 1,
        "mode":           "metronome_exercise",
        "exercise_path":  "../exercises/ex.json",
    })

    bundle = load_session_bundle(modes_dir / "session.json")

    assert bundle.practice_mode.mode   == "metronome_exercise"
    assert bundle.exercise             is not None
    assert bundle.exercise.name        == "Four Beats"
    assert bundle.alignment            is None
    assert bundle.session_log          is None


def test_load_recording_aligned_bundle(tmp_path):
    modes_dir = tmp_path / "modes"
    ex_dir    = tmp_path / "exercises"
    al_dir    = tmp_path / "alignments"

    _write_json(ex_dir / "ex.json", _exercise_json())
    _write_json(al_dir / "al.json", _alignment_json())
    _write_json(modes_dir / "session.json", {
        "schema_version": 1,
        "mode":           "recording_aligned_exercise",
        "exercise_path":  "../exercises/ex.json",
        "alignment_path": "../alignments/al.json",
    })

    bundle = load_session_bundle(modes_dir / "session.json")

    assert bundle.practice_mode.mode           == "recording_aligned_exercise"
    assert bundle.exercise                     is not None
    assert bundle.alignment                    is not None
    assert bundle.alignment.first_beat_time_sec == 2.0


def test_load_play_to_align_bundle(tmp_path):
    _write_json(tmp_path / "session.json", {
        "schema_version": 1,
        "mode":           "play_to_align",
        "audio_file":     "tracks/test.wav",
    })

    bundle = load_session_bundle(tmp_path / "session.json")

    assert bundle.practice_mode.mode == "play_to_align"
    assert bundle.exercise           is None
    assert bundle.alignment          is None


# ── 14: relative path resolution ─────────────────────────────────────────────

def test_relative_paths_resolved_from_practice_mode_directory(tmp_path):
    """Exercise path is resolved relative to the practice mode file's parent dir."""
    sub = tmp_path / "deep" / "modes"
    ex_dir = tmp_path / "deep" / "exercises"

    _write_json(ex_dir / "ex.json", _exercise_json())
    _write_json(sub / "session.json", {
        "schema_version": 1,
        "mode":           "metronome_exercise",
        "exercise_path":  "../exercises/ex.json",
    })

    bundle = load_session_bundle(sub / "session.json")

    assert bundle.exercise is not None
    assert bundle.exercise.bpm == 120.0


# ── 15–16: missing asset errors ───────────────────────────────────────────────

def test_missing_exercise_file_raises(tmp_path):
    _write_json(tmp_path / "session.json", {
        "schema_version": 1,
        "mode":           "metronome_exercise",
        "exercise_path":  "exercises/does_not_exist.json",
    })

    with pytest.raises(FileNotFoundError, match="exercise"):
        load_session_bundle(tmp_path / "session.json")


def test_missing_alignment_file_raises(tmp_path):
    ex_dir = tmp_path / "exercises"
    _write_json(ex_dir / "ex.json", _exercise_json())
    _write_json(tmp_path / "session.json", {
        "schema_version": 1,
        "mode":           "recording_aligned_exercise",
        "exercise_path":  "exercises/ex.json",
        "alignment_path": "alignments/does_not_exist.json",
    })

    with pytest.raises(FileNotFoundError, match="alignment"):
        load_session_bundle(tmp_path / "session.json")


# ── 17–19: bundle_target_audio_times ─────────────────────────────────────────

def test_metronome_target_audio_times():
    # bpm=120 → beat_s=0.5s; count_in_beats=2 → count_in_s=1.0s
    # targets at beats [0, 1, 2, 3] → [1.0, 1.5, 2.0, 2.5]
    bundle = SessionBundle(
        practice_mode = _metronome_mode(),
        exercise      = _exercise(),
    )
    times = bundle_target_audio_times(bundle)

    assert len(times) == 4
    assert times == pytest.approx([1.0, 1.5, 2.0, 2.5])


def test_aligned_target_audio_times():
    # alignment: first=2.0, last=5.0, beat_count=4 → period=1.0s
    # targets at beats [0, 1, 2, 3] → [2.0, 3.0, 4.0, 5.0]
    bundle = SessionBundle(
        practice_mode = _aligned_mode(),
        exercise      = _exercise(),
        alignment     = _alignment(),
    )
    times = bundle_target_audio_times(bundle)

    assert len(times) == 4
    assert times == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_play_to_align_target_audio_times_empty():
    bundle = SessionBundle(practice_mode=_play_to_align_mode())
    assert bundle_target_audio_times(bundle) == []
