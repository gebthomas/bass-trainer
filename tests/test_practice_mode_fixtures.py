"""Tests for practice_modes/ fixture files.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Fixture loading
  1.  basic_metronome.json loads and validates without error.
  2.  basic_recording_aligned.json loads and validates without error.
  3.  play_to_align_example.json loads and validates without error.

Fixture modes
  4.  basic_metronome mode is "metronome_exercise".
  5.  basic_recording_aligned mode is "recording_aligned_exercise".
  6.  play_to_align_example mode is "play_to_align".

required_assets — keys
  7.  basic_metronome required_assets has only "exercise_path".
  8.  basic_recording_aligned required_assets has "exercise_path" and "alignment_path".
  9.  play_to_align_example required_assets has only "audio_file".

required_assets — paths exist on disk
  10. basic_metronome exercise_path resolves to an existing file.
  11. basic_recording_aligned exercise_path resolves to an existing file.
  12. basic_recording_aligned alignment_path resolves to an existing file.

recording_aligned_exercise — asset loading
  13. Exercise referenced by basic_recording_aligned loads without error.
  14. Alignment referenced by basic_recording_aligned loads without error.

recording_aligned_exercise — exercise_target_audio_times
  15. exercise_target_audio_times produces the expected audio times:
      basic_four_beats (beats 0,1,2,3) × basic_four_beats_track (period=1.0 s,
      first=2.0 s) → [2.0, 3.0, 4.0, 5.0].
  16. Audio times list length matches exercise target count.

Save / reload round-trip
  17. Each fixture round-trips through save/load unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.alignment import exercise_target_audio_times, load_alignment_file
from core.exercise import load_exercise_file
from core.practice_mode import load_practice_mode_file, required_assets, save_practice_mode_file

_ROOT          = Path(__file__).resolve().parents[1]
_MODES_DIR     = _ROOT / "practice_modes"

_METRONOME     = _MODES_DIR / "basic_metronome.json"
_ALIGNED       = _MODES_DIR / "basic_recording_aligned.json"
_PLAY_TO_ALIGN = _MODES_DIR / "play_to_align_example.json"


# ── 1–3: Fixture loading ──────────────────────────────────────────────────────

def test_basic_metronome_loads():
    pm = load_practice_mode_file(_METRONOME)
    assert pm.schema_version == 1


def test_basic_recording_aligned_loads():
    pm = load_practice_mode_file(_ALIGNED)
    assert pm.schema_version == 1


def test_play_to_align_example_loads():
    pm = load_practice_mode_file(_PLAY_TO_ALIGN)
    assert pm.schema_version == 1


# ── 4–6: Fixture modes ────────────────────────────────────────────────────────

def test_basic_metronome_mode():
    assert load_practice_mode_file(_METRONOME).mode == "metronome_exercise"


def test_basic_recording_aligned_mode():
    assert load_practice_mode_file(_ALIGNED).mode == "recording_aligned_exercise"


def test_play_to_align_example_mode():
    assert load_practice_mode_file(_PLAY_TO_ALIGN).mode == "play_to_align"


# ── 7–9: required_assets keys ────────────────────────────────────────────────

def test_metronome_required_assets_keys():
    pm = load_practice_mode_file(_METRONOME)
    assert set(required_assets(pm).keys()) == {"exercise_path"}


def test_aligned_required_assets_keys():
    pm = load_practice_mode_file(_ALIGNED)
    assert set(required_assets(pm).keys()) == {"exercise_path", "alignment_path"}


def test_play_to_align_required_assets_keys():
    pm = load_practice_mode_file(_PLAY_TO_ALIGN)
    assert set(required_assets(pm).keys()) == {"audio_file"}


# ── 10–12: required_assets paths exist ───────────────────────────────────────

def test_metronome_exercise_path_exists():
    pm = load_practice_mode_file(_METRONOME)
    assert (_ROOT / required_assets(pm)["exercise_path"]).exists()


def test_aligned_exercise_path_exists():
    pm = load_practice_mode_file(_ALIGNED)
    assert (_ROOT / required_assets(pm)["exercise_path"]).exists()


def test_aligned_alignment_path_exists():
    pm = load_practice_mode_file(_ALIGNED)
    assert (_ROOT / required_assets(pm)["alignment_path"]).exists()


# ── 13–14: recording_aligned_exercise asset loading ──────────────────────────

def test_aligned_exercise_loads():
    pm = load_practice_mode_file(_ALIGNED)
    ex = load_exercise_file(_ROOT / pm.exercise_path)
    assert ex.name == "Basic Four Beats"


def test_aligned_alignment_loads():
    pm = load_practice_mode_file(_ALIGNED)
    align = load_alignment_file(_ROOT / pm.alignment_path)
    assert align.audio_file == "tracks/basic_four_beats.wav"


# ── 15–16: exercise_target_audio_times ───────────────────────────────────────

def test_exercise_target_audio_times_values():
    pm    = load_practice_mode_file(_ALIGNED)
    ex    = load_exercise_file(_ROOT / pm.exercise_path)
    align = load_alignment_file(_ROOT / pm.alignment_path)
    # basic_four_beats targets: beats 0,1,2,3
    # basic_four_beats_track: first=2.0, period=1.0 s
    # expected audio times: 2.0, 3.0, 4.0, 5.0
    times = exercise_target_audio_times(ex, align)
    assert times == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_exercise_target_audio_times_length():
    pm    = load_practice_mode_file(_ALIGNED)
    ex    = load_exercise_file(_ROOT / pm.exercise_path)
    align = load_alignment_file(_ROOT / pm.alignment_path)
    times = exercise_target_audio_times(ex, align)
    assert len(times) == len(ex.targets)


# ── 17: save / reload round-trip ─────────────────────────────────────────────

@pytest.mark.parametrize("fixture_path", [_METRONOME, _ALIGNED, _PLAY_TO_ALIGN])
def test_fixture_save_reload_roundtrip(fixture_path, tmp_path):
    pm   = load_practice_mode_file(fixture_path)
    dest = tmp_path / fixture_path.name
    save_practice_mode_file(pm, dest)
    pm2  = load_practice_mode_file(dest)
    assert pm == pm2
