"""Tests for core/exercise.py — versioned exercise schema.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Construction
  1.  Valid minimal exercise constructs without error.
  2.  simple_timing_exercise produces a valid Exercise.
  3.  schema_version defaults to SCHEMA_VERSION in simple_timing_exercise.

JSON round trip
  4.  exercise_to_json / exercise_from_json round-trips a minimal exercise.
  5.  Round-trip preserves description, tags, and metadata.
  6.  Round-trip preserves target label, expected_pitch, duration_beats.
  7.  exercise_to_dict / exercise_from_dict round-trip is equal to original.

Validation — Exercise
  8.  schema_version != 1 raises ValueError.
  9.  schema_version 0 raises ValueError.
  10. Empty name raises ValueError.
  11. Whitespace-only name raises ValueError.
  12. bpm == 0 raises ValueError.
  13. bpm < 0 raises ValueError.
  14. count_in_beats < 0 raises ValueError.
  15. Empty targets list raises ValueError.

Validation — Target
  16. Negative target time raises ValueError.
  17. Unsorted (decreasing) target times raise ValueError.
  18. Equal target times are allowed (nondecreasing, not strictly increasing).
  19. duration_beats == 0 raises ValueError.
  20. duration_beats < 0 raises ValueError.
  21. duration_beats > 0 is valid.

Validation — metadata
  22. Non-string metadata value on Exercise raises ValueError.
  23. Non-string metadata value on Target raises ValueError.

Fields and defaults
  24. description defaults to None.
  25. tags defaults to empty list.
  26. metadata defaults to empty dict on Exercise and Target.
  27. Metadata and tags are preserved through serialisation.

Backward compatibility
  28. Old-format {"time": ..., "note": ...} target dict is accepted.
  29. "note" field is mapped to expected_pitch.
  30. Targets with only {"time": ...} are accepted.
  31. exercise_from_dict rejects missing schema_version.
  32. exercise_from_dict rejects missing bpm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.exercise import (
    SCHEMA_VERSION,
    Exercise,
    Target,
    exercise_from_dict,
    exercise_from_json,
    exercise_to_dict,
    exercise_to_json,
    simple_timing_exercise,
    validate_exercise,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _minimal_dict() -> dict:
    """Smallest valid exercise dict."""
    return {
        "schema_version": 1,
        "name": "Test exercise",
        "bpm": 120.0,
        "count_in_beats": 0,
        "targets": [{"time": 1.0}, {"time": 2.0}],
    }


def _minimal_exercise() -> Exercise:
    return exercise_from_dict(_minimal_dict())


def _full_dict() -> dict:
    return {
        "schema_version": 1,
        "name": "Full exercise",
        "bpm": 100.0,
        "count_in_beats": 2,
        "targets": [
            {
                "time": 1.0,
                "label": "root",
                "expected_pitch": "E1",
                "duration_beats": 0.5,
                "metadata": {"difficulty": "easy"},
            },
            {
                "time": 2.0,
                "label": "fifth",
                "expected_pitch": "B1",
                "duration_beats": None,
                "metadata": {},
            },
        ],
        "description": "A two-beat exercise.",
        "tags": ["beginner", "ii-V-I"],
        "metadata": {"source": "manual", "author": "geb"},
    }


# ── 1–3: Construction ─────────────────────────────────────────────────────────

def test_minimal_exercise_valid():
    ex = _minimal_exercise()
    assert ex.name == "Test exercise"
    assert ex.bpm  == pytest.approx(120.0)
    assert len(ex.targets) == 2


def test_simple_timing_exercise_constructs():
    ex = simple_timing_exercise("Quarter notes", 120.0, 2, [1, 2, 3, 4])
    assert ex.name == "Quarter notes"
    assert len(ex.targets) == 4
    assert ex.targets[0].time == pytest.approx(1.0)
    assert ex.targets[3].time == pytest.approx(4.0)


def test_simple_timing_exercise_schema_version():
    ex = simple_timing_exercise("x", 120.0, 0, [1.0])
    assert ex.schema_version == SCHEMA_VERSION


# ── 4–7: JSON round trip ──────────────────────────────────────────────────────

def test_json_roundtrip_minimal():
    ex  = _minimal_exercise()
    ex2 = exercise_from_json(exercise_to_json(ex))
    assert ex2.name           == ex.name
    assert ex2.bpm            == pytest.approx(ex.bpm)
    assert ex2.count_in_beats == ex.count_in_beats
    assert len(ex2.targets)   == len(ex.targets)
    assert ex2.targets[0].time == pytest.approx(ex.targets[0].time)


def test_json_roundtrip_preserves_description_tags_metadata():
    ex  = exercise_from_dict(_full_dict())
    ex2 = exercise_from_json(exercise_to_json(ex))
    assert ex2.description == "A two-beat exercise."
    assert ex2.tags        == ["beginner", "ii-V-I"]
    assert ex2.metadata    == {"source": "manual", "author": "geb"}


def test_json_roundtrip_preserves_target_fields():
    ex  = exercise_from_dict(_full_dict())
    ex2 = exercise_from_json(exercise_to_json(ex))
    t   = ex2.targets[0]
    assert t.label           == "root"
    assert t.expected_pitch  == "E1"
    assert t.duration_beats  == pytest.approx(0.5)
    assert t.metadata        == {"difficulty": "easy"}


def test_dict_roundtrip_equality():
    ex  = exercise_from_dict(_full_dict())
    ex2 = exercise_from_dict(exercise_to_dict(ex))
    assert ex == ex2


# ── 8–15: Exercise-level validation ──────────────────────────────────────────

def test_schema_version_2_raises():
    data = _minimal_dict()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        exercise_from_dict(data)


def test_schema_version_0_raises():
    data = _minimal_dict()
    data["schema_version"] = 0
    with pytest.raises(ValueError, match="schema_version"):
        exercise_from_dict(data)


def test_empty_name_raises():
    data = _minimal_dict()
    data["name"] = ""
    with pytest.raises(ValueError, match="name"):
        exercise_from_dict(data)


def test_whitespace_name_raises():
    data = _minimal_dict()
    data["name"] = "   "
    with pytest.raises(ValueError, match="name"):
        exercise_from_dict(data)


def test_bpm_zero_raises():
    data = _minimal_dict()
    data["bpm"] = 0.0
    with pytest.raises(ValueError, match="bpm"):
        exercise_from_dict(data)


def test_bpm_negative_raises():
    data = _minimal_dict()
    data["bpm"] = -60.0
    with pytest.raises(ValueError, match="bpm"):
        exercise_from_dict(data)


def test_count_in_beats_negative_raises():
    data = _minimal_dict()
    data["count_in_beats"] = -1
    with pytest.raises(ValueError, match="count_in_beats"):
        exercise_from_dict(data)


def test_empty_targets_raises():
    data = _minimal_dict()
    data["targets"] = []
    with pytest.raises(ValueError, match="targets"):
        exercise_from_dict(data)


# ── 16–21: Target-level validation ────────────────────────────────────────────

def test_negative_target_time_raises():
    data = _minimal_dict()
    data["targets"] = [{"time": -0.5}]
    with pytest.raises(ValueError, match="time"):
        exercise_from_dict(data)


def test_unsorted_target_times_raises():
    data = _minimal_dict()
    data["targets"] = [{"time": 2.0}, {"time": 1.0}]
    with pytest.raises(ValueError, match="nondecreasing"):
        exercise_from_dict(data)


def test_equal_target_times_allowed():
    # Simultaneous notes — nondecreasing, not strictly increasing
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0}, {"time": 1.0}, {"time": 2.0}]
    ex = exercise_from_dict(data)
    assert ex.targets[0].time == pytest.approx(1.0)
    assert ex.targets[1].time == pytest.approx(1.0)


def test_duration_beats_zero_raises():
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0, "duration_beats": 0.0}]
    with pytest.raises(ValueError, match="duration_beats"):
        exercise_from_dict(data)


def test_duration_beats_negative_raises():
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0, "duration_beats": -0.5}]
    with pytest.raises(ValueError, match="duration_beats"):
        exercise_from_dict(data)


def test_duration_beats_positive_is_valid():
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0, "duration_beats": 0.5}]
    ex = exercise_from_dict(data)
    assert ex.targets[0].duration_beats == pytest.approx(0.5)


# ── 22–23: Metadata validation ────────────────────────────────────────────────

def test_exercise_metadata_non_string_value_raises():
    ex = Exercise(
        schema_version=1,
        name="x",
        bpm=120.0,
        count_in_beats=0,
        targets=[Target(time=1.0)],
        metadata={"key": 42},           # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_exercise(ex)


def test_target_metadata_non_string_value_raises():
    ex = Exercise(
        schema_version=1,
        name="x",
        bpm=120.0,
        count_in_beats=0,
        targets=[Target(time=1.0, metadata={"key": 3.14})],  # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_exercise(ex)


# ── 24–27: Field defaults and preservation ────────────────────────────────────

def test_description_defaults_to_none():
    ex = _minimal_exercise()
    assert ex.description is None


def test_tags_defaults_to_empty_list():
    ex = _minimal_exercise()
    assert ex.tags == []


def test_metadata_defaults_to_empty_dict():
    ex = _minimal_exercise()
    assert ex.metadata == {}
    assert ex.targets[0].metadata == {}


def test_metadata_and_tags_preserved():
    ex = exercise_from_dict(_full_dict())
    assert ex.tags                    == ["beginner", "ii-V-I"]
    assert ex.metadata["source"]      == "manual"
    assert ex.targets[0].metadata["difficulty"] == "easy"


# ── 28–32: Backward compatibility ────────────────────────────────────────────

def test_old_format_note_field_accepted():
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0, "note": "E1"}, {"time": 2.0, "note": "A1"}]
    ex = exercise_from_dict(data)
    assert len(ex.targets) == 2


def test_old_format_note_mapped_to_expected_pitch():
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0, "note": "E1"}]
    ex = exercise_from_dict(data)
    assert ex.targets[0].expected_pitch == "E1"


def test_time_only_target_dict_accepted():
    data = _minimal_dict()
    data["targets"] = [{"time": 1.0}, {"time": 2.0}, {"time": 3.0}]
    ex = exercise_from_dict(data)
    assert len(ex.targets)           == 3
    assert ex.targets[0].label       is None
    assert ex.targets[0].expected_pitch is None


def test_from_dict_missing_schema_version_raises():
    data = _minimal_dict()
    del data["schema_version"]
    with pytest.raises(KeyError):
        exercise_from_dict(data)


def test_from_dict_missing_bpm_raises():
    data = _minimal_dict()
    del data["bpm"]
    with pytest.raises(KeyError):
        exercise_from_dict(data)
