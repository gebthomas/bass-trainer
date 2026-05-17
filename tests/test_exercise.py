"""Tests for core/exercise.py.

All tests are pure Python — no audio hardware required.

Test matrix
-----------
Roundtrip
  1.  from_dict(to_dict(ex)) produces an equal Exercise.
  2.  to_dict() output is JSON-serialisable.

from_dict
  3.  Sparse "onset" sub-dict uses OnsetAdapterConfig defaults.
  4.  Sparse "tracker" sub-dict uses TrackerConfig defaults.
  5.  Absent match_window_s key → None on the object.
  6.  Present match_window_s is preserved exactly.
  7.  Partial tracker dict (only some keys) merges with defaults.

Conversion methods
  8.  to_session_engine_kwargs() contains targets, bpm, count_in_beats.
  9.  to_session_engine_kwargs() excludes match_window_s when None.
  10. to_session_engine_kwargs() includes match_window_s when set.
  11. to_onset_adapter_kwargs() excludes sample_rate.
  12. to_tracker_kwargs() excludes bpm / nominal_bpm.
  13. SessionEngine(**ex.to_session_engine_kwargs(), tracker=tracker) constructs without error.
  14. OnsetAdapter(sample_rate=48000, **ex.to_onset_adapter_kwargs()) constructs without error.
  15. TempoTracker(ex.bpm, **ex.to_tracker_kwargs()) constructs without error.
  16. Engine constructed from Exercise has the correct bpm and target count.

Validation — Exercise.__post_init__
  17. bpm <= 0 raises ValueError.
  18. count_in_beats < 0 raises ValueError.
  19. Empty targets raises ValueError.
  20. Target missing 'time' key raises ValueError.
  21. Target time <= 0 raises ValueError.
  22. Non-increasing target times raises ValueError.
  23. match_window_s = 0 raises ValueError.

Validation — OnsetAdapterConfig.__post_init__
  24. min_rms < 0 raises ValueError.
  25. min_peak < 0 raises ValueError.
  26. refractory_s = 0 raises ValueError.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.exercise import Exercise, OnsetAdapterConfig, TrackerConfig
from core.onset_adapter import OnsetAdapter
from core.session_engine import SessionEngine
from core.tempo_tracker import TempoTracker

# ── Shared test data ──────────────────────────────────────────────────────────

TARGETS_2 = [{"time": 1, "note": "E1"}, {"time": 2, "note": "A1"}]
TARGETS_4 = [
    {"time": 1, "note": "E1"},
    {"time": 2, "note": "A1"},
    {"time": 3, "note": "D2"},
    {"time": 4, "note": "G2"},
]


def _make_exercise(**overrides) -> Exercise:
    defaults = dict(name="Test", bpm=120.0, count_in_beats=0, targets=TARGETS_4)
    defaults.update(overrides)
    return Exercise(**defaults)


def _full_dict() -> dict:
    return {
        "name":           "Four Quarter Notes",
        "bpm":            120.0,
        "count_in_beats": 2,
        "targets":        TARGETS_4,
        "match_window_s": 0.25,
        "onset": {
            "min_rms":      0.020,
            "min_peak":     0.10,
            "refractory_s": 0.100,
        },
        "tracker": {
            "phase_alpha":           0.05,
            "tempo_beta":            0.20,
            "outlier_threshold":     0.35,
            "confidence_window":     6,
            "drift_window":          3,
            "drift_threshold_scale": 1.5,
            "drift_min_frac":        0.08,
        },
    }


# ── 1–2: Roundtrip ────────────────────────────────────────────────────────────

def test_roundtrip_equality():
    data = _full_dict()
    ex   = Exercise.from_dict(data)
    ex2  = Exercise.from_dict(ex.to_dict())
    assert ex == ex2


def test_to_dict_is_json_serialisable():
    ex = _make_exercise()
    json.dumps(ex.to_dict())  # must not raise


# ── 3–7: from_dict ────────────────────────────────────────────────────────────

def test_from_dict_absent_onset_uses_defaults():
    data = {"name": "x", "bpm": 100.0, "count_in_beats": 0, "targets": TARGETS_2}
    ex   = Exercise.from_dict(data)
    ref  = OnsetAdapterConfig()
    assert ex.onset.min_rms      == ref.min_rms
    assert ex.onset.min_peak     == ref.min_peak
    assert ex.onset.refractory_s == ref.refractory_s


def test_from_dict_absent_tracker_uses_defaults():
    data = {"name": "x", "bpm": 100.0, "count_in_beats": 0, "targets": TARGETS_2}
    ex   = Exercise.from_dict(data)
    ref  = TrackerConfig()
    assert ex.tracker.phase_alpha       == ref.phase_alpha
    assert ex.tracker.tempo_beta        == ref.tempo_beta
    assert ex.tracker.outlier_threshold == ref.outlier_threshold


def test_from_dict_absent_match_window_is_none():
    data = {"name": "x", "bpm": 120.0, "count_in_beats": 0, "targets": TARGETS_2}
    ex   = Exercise.from_dict(data)
    assert ex.match_window_s is None


def test_from_dict_present_match_window_preserved():
    data = {"name": "x", "bpm": 120.0, "count_in_beats": 0,
            "targets": TARGETS_2, "match_window_s": 0.20}
    ex   = Exercise.from_dict(data)
    assert ex.match_window_s == pytest.approx(0.20)


def test_from_dict_partial_tracker_merges_with_defaults():
    data = {"name": "x", "bpm": 100.0, "count_in_beats": 0, "targets": TARGETS_2,
            "tracker": {"phase_alpha": 0.05, "tempo_beta": 0.15}}
    ex   = Exercise.from_dict(data)
    assert ex.tracker.phase_alpha == pytest.approx(0.05)
    assert ex.tracker.tempo_beta  == pytest.approx(0.15)
    assert ex.tracker.drift_window == TrackerConfig().drift_window  # default unchanged


# ── 8–12: Conversion method contents ─────────────────────────────────────────

def test_session_engine_kwargs_contains_core_fields():
    ex     = _make_exercise()
    kwargs = ex.to_session_engine_kwargs()
    assert kwargs["targets"]        == ex.targets
    assert kwargs["bpm"]            == ex.bpm
    assert kwargs["count_in_beats"] == ex.count_in_beats


def test_session_engine_kwargs_excludes_match_window_when_none():
    ex = _make_exercise(match_window_s=None)
    assert "match_window_s" not in ex.to_session_engine_kwargs()


def test_session_engine_kwargs_includes_match_window_when_set():
    ex = _make_exercise(match_window_s=0.30)
    assert ex.to_session_engine_kwargs()["match_window_s"] == pytest.approx(0.30)


def test_onset_adapter_kwargs_excludes_sample_rate():
    ex     = _make_exercise()
    kwargs = ex.to_onset_adapter_kwargs()
    assert "sample_rate" not in kwargs
    assert "min_rms"      in kwargs
    assert "min_peak"     in kwargs
    assert "refractory_s" in kwargs


def test_tracker_kwargs_excludes_bpm():
    ex     = _make_exercise()
    kwargs = ex.to_tracker_kwargs()
    assert "bpm"         not in kwargs
    assert "nominal_bpm" not in kwargs
    assert "phase_alpha" in kwargs


# ── 13–16: Constructor compatibility ─────────────────────────────────────────

def test_session_engine_constructs_from_exercise():
    ex      = _make_exercise()
    tracker = TempoTracker(ex.bpm, **ex.to_tracker_kwargs())
    engine  = SessionEngine(**ex.to_session_engine_kwargs(), tracker=tracker)
    assert engine.bpm == pytest.approx(ex.bpm)
    assert len(engine.targets) == len(ex.targets)


def test_onset_adapter_constructs_from_exercise():
    ex      = _make_exercise()
    adapter = OnsetAdapter(sample_rate=48000, **ex.to_onset_adapter_kwargs())
    assert adapter.min_rms      == pytest.approx(ex.onset.min_rms)
    assert adapter.min_peak     == pytest.approx(ex.onset.min_peak)
    assert adapter.refractory_s == pytest.approx(ex.onset.refractory_s)


def test_tempo_tracker_constructs_from_exercise():
    ex      = _make_exercise()
    tracker = TempoTracker(ex.bpm, **ex.to_tracker_kwargs())
    assert tracker.nominal_beat_s == pytest.approx(60.0 / ex.bpm)


def test_engine_from_exercise_with_explicit_match_window():
    ex      = _make_exercise(match_window_s=0.20)
    tracker = TempoTracker(ex.bpm, **ex.to_tracker_kwargs())
    engine  = SessionEngine(**ex.to_session_engine_kwargs(), tracker=tracker)
    assert engine.match_window_s == pytest.approx(0.20)


# ── 17–23: Exercise validation ────────────────────────────────────────────────

def test_bpm_zero_raises():
    with pytest.raises(ValueError, match="bpm"):
        _make_exercise(bpm=0.0)


def test_bpm_negative_raises():
    with pytest.raises(ValueError, match="bpm"):
        _make_exercise(bpm=-1.0)


def test_count_in_beats_negative_raises():
    with pytest.raises(ValueError, match="count_in_beats"):
        _make_exercise(count_in_beats=-1)


def test_empty_targets_raises():
    with pytest.raises(ValueError, match="targets"):
        _make_exercise(targets=[])


def test_target_missing_time_key_raises():
    with pytest.raises(ValueError, match="'time'"):
        _make_exercise(targets=[{"note": "E1"}])


def test_target_time_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        _make_exercise(targets=[{"time": 0, "note": "E1"}])


def test_target_time_negative_raises():
    with pytest.raises(ValueError, match="positive"):
        _make_exercise(targets=[{"time": -1, "note": "E1"}])


def test_non_increasing_target_times_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_exercise(targets=[{"time": 2}, {"time": 1}])


def test_equal_target_times_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_exercise(targets=[{"time": 1}, {"time": 1}])


def test_match_window_zero_raises():
    with pytest.raises(ValueError, match="match_window_s"):
        _make_exercise(match_window_s=0.0)


def test_match_window_negative_raises():
    with pytest.raises(ValueError, match="match_window_s"):
        _make_exercise(match_window_s=-0.1)


# ── 24–26: OnsetAdapterConfig validation ─────────────────────────────────────

def test_onset_min_rms_negative_raises():
    with pytest.raises(ValueError, match="min_rms"):
        OnsetAdapterConfig(min_rms=-0.001)


def test_onset_min_peak_negative_raises():
    with pytest.raises(ValueError, match="min_peak"):
        OnsetAdapterConfig(min_peak=-0.001)


def test_onset_refractory_zero_raises():
    with pytest.raises(ValueError, match="refractory_s"):
        OnsetAdapterConfig(refractory_s=0.0)
