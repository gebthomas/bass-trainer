"""Tests for core/alignment.py — beat alignment map.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Construction
  1.  Valid minimal alignment constructs without error.
  2.  Defaults: beats_per_bar=4, confirmed_by_user=False, confidence="unknown".

BPM / period calculations
  3.  estimate_bpm_from_first_last returns correct BPM for known inputs.
  4.  beat_period_sec is correct for known alignment.
  5.  estimated_bpm matches estimate_bpm_from_first_last.
  6.  estimate_bpm_from_first_last with beat_count < 2 raises ValueError.

Beat time indexing
  7.  beat_time(align, 0) == first_beat_time_sec.
  8.  beat_time(align, beat_count - 1) == last_beat_time_sec.
  9.  beat_time returns correct intermediate times.
  10. beat_time with negative index raises ValueError.
  11. beat_times returns list of length n_beats.
  12. beat_times first element == first_beat_time_sec.
  13. beat_times last element of beat_count matches last_beat_time_sec.

JSON round-trip
  14. alignment_to_json / alignment_from_json round-trip produces equal object.
  15. metadata is preserved through JSON round-trip.
  16. confirmed_by_user=True is preserved.

Save / load round-trip
  17. save_alignment_file + load_alignment_file produces equal object.

Validation — schema_version
  18. schema_version != 1 raises ValueError.
  19. schema_version == 0 raises ValueError.

Validation — audio_file
  20. Empty audio_file raises ValueError.
  21. Whitespace-only audio_file raises ValueError.

Validation — alignment_method
  22. Unknown alignment_method raises ValueError.
  23. Each allowed method is accepted: manual_tap, player_onsets, imported.

Validation — time ordering
  24. first_beat_time_sec < 0 raises ValueError.
  25. last_beat_time_sec == first_beat_time_sec raises ValueError.
  26. last_beat_time_sec < first_beat_time_sec raises ValueError.

Validation — counts
  27. beat_count == 1 raises ValueError.
  28. beat_count == 0 raises ValueError.
  29. beats_per_bar == 0 raises ValueError.

Validation — confidence
  30. Empty confidence raises ValueError.
  31. Whitespace-only confidence raises ValueError.

Validation — metadata
  32. Non-string metadata value raises ValueError.

Serialization
  33. alignment_from_dict raises KeyError on missing required field.
  34. alignment_to_dict / alignment_from_dict round-trip equality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.alignment import (
    SCHEMA_VERSION,
    BeatAlignment,
    alignment_from_dict,
    alignment_from_json,
    alignment_to_dict,
    alignment_to_json,
    beat_period_sec,
    beat_time,
    beat_times,
    estimate_bpm_from_first_last,
    estimated_bpm,
    load_alignment_file,
    save_alignment_file,
    validate_alignment,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _minimal_dict() -> dict:
    return {
        "schema_version":      1,
        "audio_file":          "track.wav",
        "alignment_method":    "manual_tap",
        "first_beat_time_sec": 0.5,
        "last_beat_time_sec":  4.5,
        "beat_count":          9,
    }


def _minimal_alignment() -> BeatAlignment:
    return alignment_from_dict(_minimal_dict())


def _known_alignment() -> BeatAlignment:
    """120 BPM: 8 beats, first at 1.0 s, last at 4.5 s (7 × 0.5 s = 3.5 s span)."""
    return alignment_from_dict({
        "schema_version":      1,
        "audio_file":          "demo.flac",
        "alignment_method":    "player_onsets",
        "first_beat_time_sec": 1.0,
        "last_beat_time_sec":  4.5,
        "beat_count":          8,
    })


# ── 1–2: Construction ─────────────────────────────────────────────────────────

def test_minimal_alignment_valid():
    align = _minimal_alignment()
    assert align.audio_file        == "track.wav"
    assert align.alignment_method  == "manual_tap"
    assert align.beat_count        == 9


def test_defaults():
    align = _minimal_alignment()
    assert align.beats_per_bar     == 4
    assert align.confirmed_by_user is False
    assert align.confidence        == "unknown"
    assert align.metadata          == {}


# ── 3–6: BPM / period calculations ───────────────────────────────────────────

def test_estimate_bpm_known():
    # 8 beats, 7 intervals over 3.5 s → period = 0.5 s → 120 BPM
    bpm = estimate_bpm_from_first_last(1.0, 4.5, 8)
    assert bpm == pytest.approx(120.0)


def test_beat_period_known():
    align = _known_alignment()
    # (4.5 - 1.0) / (8 - 1) = 3.5 / 7 = 0.5 s
    assert beat_period_sec(align) == pytest.approx(0.5)


def test_estimated_bpm_matches_standalone():
    align = _known_alignment()
    assert estimated_bpm(align) == pytest.approx(
        estimate_bpm_from_first_last(
            align.first_beat_time_sec,
            align.last_beat_time_sec,
            align.beat_count,
        )
    )


def test_estimate_bpm_beat_count_below_2_raises():
    with pytest.raises(ValueError, match="beat_count"):
        estimate_bpm_from_first_last(0.0, 4.0, 1)


# ── 7–13: Beat time indexing ──────────────────────────────────────────────────

def test_beat_time_index_zero():
    align = _known_alignment()
    assert beat_time(align, 0) == pytest.approx(align.first_beat_time_sec)


def test_beat_time_last_index():
    align = _known_alignment()
    assert beat_time(align, align.beat_count - 1) == pytest.approx(align.last_beat_time_sec)


def test_beat_time_intermediate():
    align = _known_alignment()
    # beat 3: 1.0 + 3 * 0.5 = 2.5
    assert beat_time(align, 3) == pytest.approx(2.5)


def test_beat_time_negative_raises():
    align = _known_alignment()
    with pytest.raises(ValueError, match="beat_index"):
        beat_time(align, -1)


def test_beat_times_length():
    align = _known_alignment()
    assert len(beat_times(align, 8)) == 8


def test_beat_times_first_element():
    align = _known_alignment()
    ts = beat_times(align, 8)
    assert ts[0] == pytest.approx(align.first_beat_time_sec)


def test_beat_times_last_of_beat_count():
    align = _known_alignment()
    ts = beat_times(align, align.beat_count)
    assert ts[-1] == pytest.approx(align.last_beat_time_sec)


# ── 14–16: JSON round-trip ────────────────────────────────────────────────────

def test_json_roundtrip_equality():
    align  = _known_alignment()
    align2 = alignment_from_json(alignment_to_json(align))
    assert align == align2


def test_json_roundtrip_preserves_metadata():
    data = _minimal_dict()
    data["metadata"] = {"session": "2026-05-23", "recorder": "Geb"}
    align  = alignment_from_dict(data)
    align2 = alignment_from_json(alignment_to_json(align))
    assert align2.metadata == {"session": "2026-05-23", "recorder": "Geb"}


def test_json_roundtrip_preserves_confirmed_by_user():
    data = _minimal_dict()
    data["confirmed_by_user"] = True
    align  = alignment_from_dict(data)
    align2 = alignment_from_json(alignment_to_json(align))
    assert align2.confirmed_by_user is True


# ── 17: Save / load round-trip ────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    align = _known_alignment()
    dest  = tmp_path / "align.json"
    save_alignment_file(align, dest)
    align2 = load_alignment_file(dest)
    assert align == align2


# ── 18–19: schema_version ────────────────────────────────────────────────────

def test_schema_version_2_raises():
    data = _minimal_dict()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        alignment_from_dict(data)


def test_schema_version_0_raises():
    data = _minimal_dict()
    data["schema_version"] = 0
    with pytest.raises(ValueError, match="schema_version"):
        alignment_from_dict(data)


# ── 20–21: audio_file ────────────────────────────────────────────────────────

def test_empty_audio_file_raises():
    data = _minimal_dict()
    data["audio_file"] = ""
    with pytest.raises(ValueError, match="audio_file"):
        alignment_from_dict(data)


def test_whitespace_audio_file_raises():
    data = _minimal_dict()
    data["audio_file"] = "   "
    with pytest.raises(ValueError, match="audio_file"):
        alignment_from_dict(data)


# ── 22–23: alignment_method ──────────────────────────────────────────────────

def test_unknown_alignment_method_raises():
    data = _minimal_dict()
    data["alignment_method"] = "ai_magic"
    with pytest.raises(ValueError, match="alignment_method"):
        alignment_from_dict(data)


@pytest.mark.parametrize("method", ["manual_tap", "player_onsets", "imported"])
def test_all_allowed_methods_accepted(method):
    data = _minimal_dict()
    data["alignment_method"] = method
    align = alignment_from_dict(data)
    assert align.alignment_method == method


# ── 24–26: time ordering ──────────────────────────────────────────────────────

def test_negative_first_beat_raises():
    data = _minimal_dict()
    data["first_beat_time_sec"] = -0.1
    with pytest.raises(ValueError, match="first_beat_time_sec"):
        alignment_from_dict(data)


def test_equal_times_raises():
    data = _minimal_dict()
    data["last_beat_time_sec"] = data["first_beat_time_sec"]
    with pytest.raises(ValueError, match="last_beat_time_sec"):
        alignment_from_dict(data)


def test_reversed_times_raises():
    data = _minimal_dict()
    data["first_beat_time_sec"] = 5.0
    data["last_beat_time_sec"]  = 2.0
    with pytest.raises(ValueError, match="last_beat_time_sec"):
        alignment_from_dict(data)


# ── 27–29: counts ────────────────────────────────────────────────────────────

def test_beat_count_1_raises():
    data = _minimal_dict()
    data["beat_count"] = 1
    with pytest.raises(ValueError, match="beat_count"):
        alignment_from_dict(data)


def test_beat_count_0_raises():
    data = _minimal_dict()
    data["beat_count"] = 0
    with pytest.raises(ValueError, match="beat_count"):
        alignment_from_dict(data)


def test_beats_per_bar_0_raises():
    data = _minimal_dict()
    data["beats_per_bar"] = 0
    with pytest.raises(ValueError, match="beats_per_bar"):
        alignment_from_dict(data)


# ── 30–31: confidence ────────────────────────────────────────────────────────

def test_empty_confidence_raises():
    align = BeatAlignment(
        schema_version=1,
        audio_file="track.wav",
        alignment_method="manual_tap",
        first_beat_time_sec=0.5,
        last_beat_time_sec=4.5,
        beat_count=9,
        confidence="",
    )
    with pytest.raises(ValueError, match="confidence"):
        validate_alignment(align)


def test_whitespace_confidence_raises():
    align = BeatAlignment(
        schema_version=1,
        audio_file="track.wav",
        alignment_method="manual_tap",
        first_beat_time_sec=0.5,
        last_beat_time_sec=4.5,
        beat_count=9,
        confidence="   ",
    )
    with pytest.raises(ValueError, match="confidence"):
        validate_alignment(align)


# ── 32: metadata ─────────────────────────────────────────────────────────────

def test_metadata_non_string_value_raises():
    align = BeatAlignment(
        schema_version=1,
        audio_file="track.wav",
        alignment_method="manual_tap",
        first_beat_time_sec=0.5,
        last_beat_time_sec=4.5,
        beat_count=9,
        metadata={"key": 42},           # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_alignment(align)


# ── 33–34: serialisation ─────────────────────────────────────────────────────

def test_from_dict_missing_required_field_raises():
    data = _minimal_dict()
    del data["beat_count"]
    with pytest.raises(KeyError):
        alignment_from_dict(data)


def test_dict_roundtrip_equality():
    align  = _known_alignment()
    align2 = alignment_from_dict(alignment_to_dict(align))
    assert align == align2


# ── 35–39: Time signatures — beats_per_bar independence ──────────────────────
#
# beats_per_bar is for display / bar-counting only.  It must not influence
# beat_time() or estimated_bpm().  Tests confirm 3/4 and 5/4 alignments are
# valid and that changing beats_per_bar on an otherwise identical alignment
# leaves the beat grid unchanged.

@pytest.mark.parametrize("beats_per_bar", [3, 5])
def test_non_4_4_time_signature_validates(beats_per_bar):
    """3/4 and 5/4 alignments must pass validation."""
    data = _minimal_dict()
    data["beats_per_bar"] = beats_per_bar
    align = alignment_from_dict(data)
    assert align.beats_per_bar == beats_per_bar


def test_3_4_beat_times_match_4_4():
    """beat_time() is independent of beats_per_bar."""
    base = _minimal_dict()
    a4   = alignment_from_dict({**base, "beats_per_bar": 4})
    a3   = alignment_from_dict({**base, "beats_per_bar": 3})
    for i in range(a4.beat_count):
        assert beat_time(a4, i) == pytest.approx(beat_time(a3, i))


def test_5_4_beat_times_match_4_4():
    """beat_time() is independent of beats_per_bar."""
    base = _minimal_dict()
    a4   = alignment_from_dict({**base, "beats_per_bar": 4})
    a5   = alignment_from_dict({**base, "beats_per_bar": 5})
    for i in range(a4.beat_count):
        assert beat_time(a4, i) == pytest.approx(beat_time(a5, i))


def test_estimated_bpm_independent_of_beats_per_bar():
    """estimated_bpm() is independent of beats_per_bar."""
    base = _minimal_dict()
    bpms = [
        estimated_bpm(alignment_from_dict({**base, "beats_per_bar": bpb}))
        for bpb in (2, 3, 4, 5, 6, 7)
    ]
    assert all(b == pytest.approx(bpms[0]) for b in bpms)


def test_3_4_waltz_alignment():
    """A concrete 3/4 waltz alignment round-trips and has correct BPM."""
    data = {
        "schema_version":      1,
        "audio_file":          "waltz.wav",
        "alignment_method":    "manual_tap",
        "first_beat_time_sec": 0.5,
        "last_beat_time_sec":  5.3,  # 9 beats of a waltz @ ~100 BPM
        "beat_count":          9,
        "beats_per_bar":       3,
        "confirmed_by_user":   False,
        "confidence":          "low",
    }
    align = alignment_from_dict(data)
    assert align.beats_per_bar == 3
    # period = (5.3 - 0.5) / 8 = 0.6 s → 100 BPM
    assert estimated_bpm(align) == pytest.approx(100.0)
    # round-trip preserves beats_per_bar
    align2 = alignment_from_dict(alignment_to_dict(align))
    assert align2.beats_per_bar == 3
