"""Tests for core/practice_mode.py — practice session configuration models.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Construction
  1.  Valid metronome_exercise constructs without error.
  2.  Valid recording_aligned_exercise constructs without error.
  3.  Valid play_to_align constructs without error.
  4.  Defaults: exercise_path, alignment_path, audio_file, description all None;
      metadata empty dict.

Validation — mode
  5.  Unknown mode raises ValueError.
  6.  Each allowed mode is accepted: metronome_exercise, recording_aligned_exercise,
      play_to_align.

Validation — schema_version
  7.  schema_version != 1 raises ValueError.

Validation — metronome_exercise requirements
  8.  Missing exercise_path raises ValueError.
  9.  Whitespace-only exercise_path raises ValueError.

Validation — recording_aligned_exercise requirements
  10. Missing exercise_path raises ValueError.
  11. Missing alignment_path raises ValueError.
  12. Whitespace-only alignment_path raises ValueError.

Validation — play_to_align requirements
  13. Missing audio_file raises ValueError.
  14. Whitespace-only audio_file raises ValueError.

Validation — metadata
  15. Non-string metadata value raises ValueError.

Serialization
  16. practice_mode_from_dict raises KeyError on missing schema_version.
  17. practice_mode_from_dict raises KeyError on missing mode.
  18. practice_mode_to_dict / practice_mode_from_dict round-trip equality.
  19. JSON round-trip produces equal object.
  20. Optional fields preserved through JSON round-trip.

File I/O
  21. save_practice_mode_file + load_practice_mode_file round-trip equality.

Metadata
  22. Metadata preserved through serialization.

required_assets
  23. metronome_exercise returns {"exercise_path": ...}.
  24. recording_aligned_exercise returns {"exercise_path": ..., "alignment_path": ...}.
  25. play_to_align returns {"audio_file": ...}.
  26. required_assets keys contain correct values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.practice_mode import (
    SCHEMA_VERSION,
    PracticeMode,
    load_practice_mode_file,
    practice_mode_from_dict,
    practice_mode_from_json,
    practice_mode_to_dict,
    practice_mode_to_json,
    required_assets,
    save_practice_mode_file,
    validate_practice_mode,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _metronome_dict() -> dict:
    return {
        "schema_version": 1,
        "mode":           "metronome_exercise",
        "exercise_path":  "exercises/basic_four_beats.json",
    }


def _aligned_dict() -> dict:
    return {
        "schema_version": 1,
        "mode":           "recording_aligned_exercise",
        "exercise_path":  "exercises/basic_four_beats.json",
        "alignment_path": "alignments/basic_four_beats_track.json",
    }


def _play_to_align_dict() -> dict:
    return {
        "schema_version": 1,
        "mode":           "play_to_align",
        "audio_file":     "tracks/my_recording.wav",
    }


# ── 1–4: Construction ─────────────────────────────────────────────────────────

def test_metronome_exercise_valid():
    pm = practice_mode_from_dict(_metronome_dict())
    assert pm.mode          == "metronome_exercise"
    assert pm.exercise_path == "exercises/basic_four_beats.json"


def test_recording_aligned_exercise_valid():
    pm = practice_mode_from_dict(_aligned_dict())
    assert pm.mode           == "recording_aligned_exercise"
    assert pm.exercise_path  == "exercises/basic_four_beats.json"
    assert pm.alignment_path == "alignments/basic_four_beats_track.json"


def test_play_to_align_valid():
    pm = practice_mode_from_dict(_play_to_align_dict())
    assert pm.mode       == "play_to_align"
    assert pm.audio_file == "tracks/my_recording.wav"


def test_defaults():
    pm = practice_mode_from_dict(_metronome_dict())
    assert pm.alignment_path is None
    assert pm.audio_file     is None
    assert pm.description    is None
    assert pm.metadata       == {}


# ── 5–6: mode validation ──────────────────────────────────────────────────────

def test_unknown_mode_raises():
    data = _metronome_dict()
    data["mode"] = "free_jam"
    with pytest.raises(ValueError, match="mode"):
        practice_mode_from_dict(data)


@pytest.mark.parametrize("mode,extra", [
    ("metronome_exercise",           {"exercise_path": "ex.json"}),
    ("recording_aligned_exercise",   {"exercise_path": "ex.json", "alignment_path": "al.json"}),
    ("play_to_align",                {"audio_file": "track.wav"}),
])
def test_all_allowed_modes_accepted(mode, extra):
    data = {"schema_version": 1, "mode": mode, **extra}
    pm = practice_mode_from_dict(data)
    assert pm.mode == mode


# ── 7: schema_version ────────────────────────────────────────────────────────

def test_schema_version_2_raises():
    data = _metronome_dict()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        practice_mode_from_dict(data)


# ── 8–9: metronome_exercise requirements ─────────────────────────────────────

def test_metronome_missing_exercise_path_raises():
    data = _metronome_dict()
    del data["exercise_path"]
    with pytest.raises(ValueError, match="exercise_path"):
        practice_mode_from_dict(data)


def test_metronome_whitespace_exercise_path_raises():
    data = _metronome_dict()
    data["exercise_path"] = "   "
    with pytest.raises(ValueError, match="exercise_path"):
        practice_mode_from_dict(data)


# ── 10–12: recording_aligned_exercise requirements ───────────────────────────

def test_aligned_missing_exercise_path_raises():
    data = _aligned_dict()
    del data["exercise_path"]
    with pytest.raises(ValueError, match="exercise_path"):
        practice_mode_from_dict(data)


def test_aligned_missing_alignment_path_raises():
    data = _aligned_dict()
    del data["alignment_path"]
    with pytest.raises(ValueError, match="alignment_path"):
        practice_mode_from_dict(data)


def test_aligned_whitespace_alignment_path_raises():
    data = _aligned_dict()
    data["alignment_path"] = "   "
    with pytest.raises(ValueError, match="alignment_path"):
        practice_mode_from_dict(data)


# ── 13–14: play_to_align requirements ────────────────────────────────────────

def test_play_to_align_missing_audio_file_raises():
    data = _play_to_align_dict()
    del data["audio_file"]
    with pytest.raises(ValueError, match="audio_file"):
        practice_mode_from_dict(data)


def test_play_to_align_whitespace_audio_file_raises():
    data = _play_to_align_dict()
    data["audio_file"] = "   "
    with pytest.raises(ValueError, match="audio_file"):
        practice_mode_from_dict(data)


# ── 15: metadata validation ───────────────────────────────────────────────────

def test_metadata_non_string_value_raises():
    pm = PracticeMode(
        schema_version=1,
        mode="metronome_exercise",
        exercise_path="ex.json",
        metadata={"level": 3},          # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_practice_mode(pm)


# ── 16–20: serialization ─────────────────────────────────────────────────────

def test_from_dict_missing_schema_version_raises():
    data = _metronome_dict()
    del data["schema_version"]
    with pytest.raises(KeyError):
        practice_mode_from_dict(data)


def test_from_dict_missing_mode_raises():
    data = _metronome_dict()
    del data["mode"]
    with pytest.raises(KeyError):
        practice_mode_from_dict(data)


def test_dict_roundtrip_equality():
    pm  = practice_mode_from_dict(_aligned_dict())
    pm2 = practice_mode_from_dict(practice_mode_to_dict(pm))
    assert pm == pm2


def test_json_roundtrip_equality():
    pm  = practice_mode_from_dict(_aligned_dict())
    pm2 = practice_mode_from_json(practice_mode_to_json(pm))
    assert pm == pm2


def test_optional_fields_preserved_through_json():
    data = _aligned_dict()
    data["description"] = "Practice walking bass over Autumn Leaves."
    data["audio_file"]  = "tracks/autumn_leaves.wav"
    pm  = practice_mode_from_dict(data)
    pm2 = practice_mode_from_json(practice_mode_to_json(pm))
    assert pm2.description == "Practice walking bass over Autumn Leaves."
    assert pm2.audio_file  == "tracks/autumn_leaves.wav"


# ── 21: file I/O ──────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    pm   = practice_mode_from_dict(_aligned_dict())
    dest = tmp_path / "session.practice_mode.json"
    save_practice_mode_file(pm, dest)
    pm2  = load_practice_mode_file(dest)
    assert pm == pm2


# ── 22: metadata preservation ────────────────────────────────────────────────

def test_metadata_preserved():
    data = _metronome_dict()
    data["metadata"] = {"tempo_feel": "relaxed", "session_date": "2026-05-23"}
    pm  = practice_mode_from_dict(data)
    pm2 = practice_mode_from_json(practice_mode_to_json(pm))
    assert pm2.metadata == {"tempo_feel": "relaxed", "session_date": "2026-05-23"}


# ── 23–26: required_assets ────────────────────────────────────────────────────

def test_required_assets_metronome():
    pm = practice_mode_from_dict(_metronome_dict())
    assets = required_assets(pm)
    assert set(assets.keys()) == {"exercise_path"}


def test_required_assets_aligned():
    pm = practice_mode_from_dict(_aligned_dict())
    assets = required_assets(pm)
    assert set(assets.keys()) == {"exercise_path", "alignment_path"}


def test_required_assets_play_to_align():
    pm = practice_mode_from_dict(_play_to_align_dict())
    assets = required_assets(pm)
    assert set(assets.keys()) == {"audio_file"}


def test_required_assets_values_correct():
    pm = practice_mode_from_dict(_aligned_dict())
    assets = required_assets(pm)
    assert assets["exercise_path"]  == "exercises/basic_four_beats.json"
    assert assets["alignment_path"] == "alignments/basic_four_beats_track.json"
