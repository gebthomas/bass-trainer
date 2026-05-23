"""Integration tests: Exercise objects in the headless session/replay path.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Adapter — exercise_targets
  1.  exercise_targets(Exercise) returns list[dict] with correct "time" values.
  2.  exercise_targets(Exercise) maps expected_pitch to "note".
  3.  exercise_targets(Exercise) includes "label" when set.
  4.  exercise_targets(Exercise) includes "duration_beats" when set.
  5.  exercise_targets(Exercise) includes "metadata" when non-empty.
  6.  exercise_targets(Exercise) omits optional keys when all are None / empty.
  7.  exercise_targets(list[dict]) returns a copy of the list unchanged.

Adapter — exercise_bpm
  8.  exercise_bpm(Exercise) returns exercise.bpm.
  9.  exercise_bpm(list, fallback_bpm=X) returns X.
  10. exercise_bpm(None, fallback_bpm=X) returns X.
  11. exercise_bpm(list, fallback_bpm=None) raises ValueError.

Adapter — exercise_count_in_beats
  12. exercise_count_in_beats(Exercise) returns exercise.count_in_beats.
  13. exercise_count_in_beats(list, fallback=X) returns X.
  14. exercise_count_in_beats(None, fallback=None) raises ValueError.

SessionEngine integration
  15. Exercise targets work with SessionEngine.on_onset (hit events emitted).
  16. Exercise targets work with SessionEngine.update_time (miss events emitted).

ready_targets (live_feedback) integration
  17. exercise_targets output works with ready_targets from live_feedback.

replay_session_data integration
  18. Exercise targets work with replay_session_data (Exercise-derived data dict).
  19. replay total and hit count match expectations.

Raw dict passthrough
  20. Raw target dicts still produce the same results from SessionEngine as before.
  21. Raw target dicts still work in replay_session_data.

End-to-end fixture
  22. basic_four_beats.json loaded and replayed with four synthetic onsets.
  23. simple_pitch_targets.json: "note" field set correctly in engine dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.exercise import (
    Exercise,
    Target,
    exercise_bpm,
    exercise_count_in_beats,
    exercise_targets,
    load_exercise_file,
    simple_timing_exercise,
)
from core.live_feedback import ready_targets
from core.session_engine import SessionEngine
from core.session_replay import replay_session_data
from core.tempo_tracker import TempoTracker


# ── Shared helpers ────────────────────────────────────────────────────────────

_EXERCISES_DIR = Path(__file__).resolve().parents[1] / "exercises"


def _make_exercise(
    times: list[float] = (1.0, 2.0, 3.0, 4.0),
    bpm: float = 120.0,
    count_in: int = 0,
) -> Exercise:
    return simple_timing_exercise("test", bpm, count_in, list(times))


def _engine_from_exercise(ex: Exercise, **kw) -> SessionEngine:
    return SessionEngine(
        exercise_targets(ex),
        bpm=exercise_bpm(ex),
        count_in_beats=exercise_count_in_beats(ex),
        **kw,
    )


# ── 1–7: exercise_targets ────────────────────────────────────────────────────

def test_exercise_targets_time_values():
    ex = _make_exercise([1.0, 2.0, 3.0])
    dicts = exercise_targets(ex)
    assert [d["time"] for d in dicts] == pytest.approx([1.0, 2.0, 3.0])


def test_exercise_targets_maps_expected_pitch_to_note():
    ex = Exercise(
        schema_version=1, name="x", bpm=120.0, count_in_beats=0,
        targets=[Target(time=1.0, expected_pitch="E2")],
    )
    dicts = exercise_targets(ex)
    assert dicts[0]["note"] == "E2"


def test_exercise_targets_includes_label():
    ex = Exercise(
        schema_version=1, name="x", bpm=120.0, count_in_beats=0,
        targets=[Target(time=1.0, label="root")],
    )
    dicts = exercise_targets(ex)
    assert dicts[0]["label"] == "root"


def test_exercise_targets_includes_duration_beats():
    ex = Exercise(
        schema_version=1, name="x", bpm=120.0, count_in_beats=0,
        targets=[Target(time=1.0, duration_beats=0.5)],
    )
    dicts = exercise_targets(ex)
    assert dicts[0]["duration_beats"] == pytest.approx(0.5)


def test_exercise_targets_includes_metadata():
    ex = Exercise(
        schema_version=1, name="x", bpm=120.0, count_in_beats=0,
        targets=[Target(time=1.0, metadata={"difficulty": "easy"})],
    )
    dicts = exercise_targets(ex)
    assert dicts[0]["metadata"] == {"difficulty": "easy"}


def test_exercise_targets_omits_none_optional_keys():
    ex = _make_exercise([1.0])
    d = exercise_targets(ex)[0]
    assert "note"           not in d
    assert "label"          not in d
    assert "duration_beats" not in d
    assert "metadata"       not in d


def test_exercise_targets_passthrough_raw_list():
    raw = [{"time": 1.0}, {"time": 2.0}]
    result = exercise_targets(raw)
    assert result == raw
    assert result is not raw  # shallow copy


# ── 8–11: exercise_bpm ───────────────────────────────────────────────────────

def test_exercise_bpm_from_exercise():
    ex = _make_exercise(bpm=100.0)
    assert exercise_bpm(ex) == pytest.approx(100.0)


def test_exercise_bpm_fallback_from_list():
    assert exercise_bpm([{"time": 1.0}], fallback_bpm=80.0) == pytest.approx(80.0)


def test_exercise_bpm_fallback_from_none():
    assert exercise_bpm(None, fallback_bpm=72.0) == pytest.approx(72.0)


def test_exercise_bpm_raises_without_fallback():
    with pytest.raises(ValueError, match="fallback_bpm"):
        exercise_bpm([{"time": 1.0}])


# ── 12–14: exercise_count_in_beats ───────────────────────────────────────────

def test_exercise_count_in_beats_from_exercise():
    ex = _make_exercise(count_in=2)
    assert exercise_count_in_beats(ex) == 2


def test_exercise_count_in_beats_fallback():
    assert exercise_count_in_beats([{"time": 1.0}], fallback_count_in_beats=4) == 4


def test_exercise_count_in_beats_raises_without_fallback():
    with pytest.raises(ValueError, match="fallback_count_in_beats"):
        exercise_count_in_beats(None)


# ── 15–16: SessionEngine integration ─────────────────────────────────────────

def test_session_engine_hit_from_exercise():
    ex = _make_exercise([1.0, 2.0], bpm=120.0, count_in=0)
    engine = _engine_from_exercise(ex)
    # nominal time for target 0 = 0 + 1.0 * 0.5 = 0.5 s
    events = engine.on_onset(0.505)
    assert len(events) == 1
    assert events[0]["severity"] in {"good", "warn"}
    assert events[0]["detected_note"] == "?"


def test_session_engine_miss_from_exercise():
    ex = _make_exercise([1.0], bpm=120.0, count_in=0)
    engine = _engine_from_exercise(ex)
    # Advance past the deadline without firing an onset
    deadline = 0.5 + engine.match_window_s + 0.01
    events = engine.update_time(deadline)
    assert len(events) == 1
    assert events[0]["severity"] == "miss"
    assert events[0]["detected_note"] is None


# ── 17: ready_targets integration ────────────────────────────────────────────

def test_ready_targets_with_exercise_derived_dicts():
    ex = _make_exercise([1.0, 2.0, 3.0, 4.0], bpm=120.0, count_in=2)
    targets = exercise_targets(ex)
    bpm = exercise_bpm(ex)
    count_in = exercise_count_in_beats(ex)
    sample_rate = 48_000

    # count-in = 2 beats @ 120 BPM = 1.0 s; target 0 at beat 1.0 → audio time 1.5 s
    # gap = 0.5 s → window_end = 1.5 + min(0.35, 0.3) = 1.8 s; threshold = 1.8 + 0.15 = 1.95 s
    current_sample = round(2.0 * sample_rate)
    indices = ready_targets(targets, bpm, count_in, sample_rate, current_sample, set())
    assert 0 in indices


# ── 18–19: replay_session_data integration ───────────────────────────────────

def _replay_data_from_exercise(ex: Exercise, onsets: list[float]) -> dict:
    return {
        "bpm":            exercise_bpm(ex),
        "count_in_beats": exercise_count_in_beats(ex),
        "targets":        exercise_targets(ex),
        "onsets":         onsets,
    }


def test_replay_session_data_from_exercise():
    ex = _make_exercise([1.0, 2.0, 3.0, 4.0], bpm=120.0, count_in=0)
    # beat_s = 0.5 s; target nominals are 0.5, 1.0, 1.5, 2.0 s
    onsets = [0.505, 1.005, 1.505, 2.005]
    data = _replay_data_from_exercise(ex, onsets)
    events = replay_session_data(data)
    assert len(events) == 4


def test_replay_hit_count_from_exercise():
    ex = _make_exercise([1.0, 2.0, 3.0, 4.0], bpm=120.0, count_in=0)
    onsets = [0.505, 1.005, 1.505, 2.005]
    data = _replay_data_from_exercise(ex, onsets)
    events = replay_session_data(data)
    hits = [e for e in events if e["detected_note"] is not None]
    assert len(hits) == 4


# ── 20–21: raw dict passthrough ──────────────────────────────────────────────

def test_raw_dicts_still_work_in_session_engine():
    targets = [{"time": 1.0}, {"time": 2.0}]
    engine = SessionEngine(targets, bpm=120.0, count_in_beats=0)
    events = engine.on_onset(0.505)
    assert len(events) == 1
    assert events[0]["detected_note"] == "?"


def test_raw_dicts_still_work_in_replay():
    data = {
        "bpm": 120.0,
        "count_in_beats": 0,
        "targets": [{"time": 1.0}, {"time": 2.0}, {"time": 3.0}, {"time": 4.0}],
        "onsets": [0.505, 1.005, 1.505, 2.005],
    }
    events = replay_session_data(data)
    assert len(events) == 4


# ── 22–23: end-to-end fixture ─────────────────────────────────────────────────

def test_basic_four_beats_replay():
    ex = load_exercise_file(_EXERCISES_DIR / "basic_four_beats.json")
    # bpm=120, count_in=2: beat_s=0.5, count_in_s=1.0
    # targets at beats 0,1,2,3 → audio times 1.0, 1.5, 2.0, 2.5 s
    onsets = [1.005, 1.505, 2.005, 2.505]
    data = _replay_data_from_exercise(ex, onsets)
    events = replay_session_data(data)
    assert len(events) == 4
    hits = [e for e in events if e["detected_note"] is not None]
    assert len(hits) == 4


def test_simple_pitch_targets_note_field():
    ex = load_exercise_file(_EXERCISES_DIR / "simple_pitch_targets.json")
    dicts = exercise_targets(ex)
    pitches = [d.get("note") for d in dicts]
    assert pitches == ["E2", "A2", "D3", "G3"]
