"""Tests for load_exercise_file / save_exercise_file in core/exercise.py.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Fixture loading
  1.  basic_four_beats.json loads without error.
  2.  eighth_notes_120.json loads without error.
  3.  simple_pitch_targets.json loads without error.

Fixture content
  4.  basic_four_beats: bpm=120, count_in=2, 4 targets, tags include "beginner".
  5.  eighth_notes_120: bpm=120, 8 targets, first time=0, last time=3.5.
  6.  simple_pitch_targets: bpm=80, 4 targets, expected_pitch values are set.

Save / reload round-trip
  7.  save_exercise_file + load_exercise_file round-trips to an equal Exercise.
  8.  Saved file is valid JSON that can be read back via exercise_from_json.

Error cases
  9.  load_exercise_file on a missing file raises FileNotFoundError.
  10. load_exercise_file on a file with invalid JSON raises ValueError or
      json.JSONDecodeError (both are acceptable parsing failures).
  11. load_exercise_file on a file with valid JSON but invalid exercise data
      raises ValueError.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.exercise import (
    Exercise,
    Target,
    exercise_from_json,
    load_exercise_file,
    save_exercise_file,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_EXERCISES_DIR = Path(__file__).resolve().parents[1] / "exercises"
_BASIC         = _EXERCISES_DIR / "basic_four_beats.json"
_EIGHTH        = _EXERCISES_DIR / "eighth_notes_120.json"
_PITCH         = _EXERCISES_DIR / "simple_pitch_targets.json"


# ── 1–3: Fixture loading ──────────────────────────────────────────────────────

def test_basic_four_beats_loads():
    ex = load_exercise_file(_BASIC)
    assert ex.name == "Basic Four Beats"


def test_eighth_notes_120_loads():
    ex = load_exercise_file(_EIGHTH)
    assert ex.name == "Eighth Notes at 120"


def test_simple_pitch_targets_loads():
    ex = load_exercise_file(_PITCH)
    assert ex.name == "Open String Pitches"


# ── 4–6: Fixture content ──────────────────────────────────────────────────────

def test_basic_four_beats_content():
    ex = load_exercise_file(_BASIC)
    assert ex.bpm            == pytest.approx(120.0)
    assert ex.count_in_beats == 2
    assert len(ex.targets)   == 4
    assert "beginner" in ex.tags


def test_eighth_notes_120_content():
    ex = load_exercise_file(_EIGHTH)
    assert ex.bpm          == pytest.approx(120.0)
    assert len(ex.targets) == 8
    assert ex.targets[0].time == pytest.approx(0.0)
    assert ex.targets[7].time == pytest.approx(3.5)


def test_simple_pitch_targets_content():
    ex = load_exercise_file(_PITCH)
    assert ex.bpm          == pytest.approx(80.0)
    assert len(ex.targets) == 4
    pitches = [t.expected_pitch for t in ex.targets]
    assert pitches == ["E2", "A2", "D3", "G3"]


# ── 7–8: Save / reload round-trip ────────────────────────────────────────────

def test_save_reload_roundtrip(tmp_path):
    ex   = load_exercise_file(_BASIC)
    dest = tmp_path / "out.json"
    save_exercise_file(ex, dest)
    ex2  = load_exercise_file(dest)
    assert ex == ex2


def test_saved_file_is_valid_json(tmp_path):
    ex   = load_exercise_file(_PITCH)
    dest = tmp_path / "pitch_out.json"
    save_exercise_file(ex, dest)
    text = dest.read_text(encoding="utf-8")
    data = json.loads(text)             # must not raise
    ex2  = exercise_from_json(text)     # must not raise
    assert ex2.bpm == pytest.approx(ex.bpm)


# ── 9–11: Error cases ────────────────────────────────────────────────────────

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_exercise_file(tmp_path / "does_not_exist.json")


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json }", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_exercise_file(bad)


def test_invalid_exercise_data_raises(tmp_path):
    bad = tmp_path / "invalid.json"
    bad.write_text(
        json.dumps({"schema_version": 1, "name": "", "bpm": 120.0,
                    "count_in_beats": 0, "targets": [{"time": 1.0}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="name"):
        load_exercise_file(bad)
