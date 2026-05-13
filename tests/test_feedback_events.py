"""Tests for core/feedback_events.py — pure feedback logic, no audio hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.feedback_events import feedback_event, summarize_feedback


# ── Fixtures ──────────────────────────────────────────────────────────────────

_TARGET = {"time": 0, "note": "D2", "string": "D", "finger": 0}


def _event(timing=0.0, pitch=0.0, detected="D2", confidence=None):
    """Build a feedback event with common defaults."""
    result: dict = {
        "detected_note":     detected,
        "timing_error_s":    timing,
        "pitch_error_cents": pitch,
    }
    if confidence is not None:
        result["confidence"] = confidence
    return feedback_event(0, _TARGET, result)


# ── Output structure ──────────────────────────────────────────────────────────

def test_event_contains_all_required_keys():
    e = _event()
    for key in ("target_idx", "expected_note", "detected_note",
                "timing_error_s", "pitch_error_cents", "confidence",
                "severity", "messages"):
        assert key in e, f"missing key: {key!r}"


def test_event_carries_target_idx():
    e = feedback_event(7, _TARGET, {"detected_note": "D2", "timing_error_s": 0.0,
                                    "pitch_error_cents": 0.0})
    assert e["target_idx"] == 7


def test_event_carries_expected_note_from_target():
    e = _event()
    assert e["expected_note"] == "D2"


def test_event_carries_detected_note():
    e = feedback_event(0, _TARGET, {"detected_note": "F2", "timing_error_s": 0.0,
                                    "pitch_error_cents": 0.0})
    assert e["detected_note"] == "F2"


# ── Good timing / good pitch ──────────────────────────────────────────────────

def test_perfect_timing_and_pitch():
    e = _event(timing=0.0, pitch=0.0)
    assert e["severity"] == "good"
    assert "Good timing" in e["messages"]


def test_good_timing_good_pitch():
    e = _event(timing=0.02, pitch=10.0)
    assert e["severity"] == "good"
    assert "Good timing" in e["messages"]


def test_good_timing_negative_values():
    e = _event(timing=-0.02, pitch=-5.0)
    assert e["severity"] == "good"
    assert "Good timing" in e["messages"]


def test_timing_exactly_at_good_boundary():
    # |0.05| <= 0.05 → good
    assert _event(timing=0.05,  pitch=0.0)["severity"] == "good"
    assert _event(timing=-0.05, pitch=0.0)["severity"] == "good"


def test_timing_just_past_good_boundary():
    e = _event(timing=0.051, pitch=0.0)
    assert e["severity"] == "warn"
    assert "A little late" in e["messages"]


def test_pitch_exactly_at_good_boundary_not_flagged():
    # |25| <= 25 → good, no pitch message
    e_high = _event(timing=0.0, pitch=25.0)
    e_low  = _event(timing=0.0, pitch=-25.0)
    assert e_high["severity"] == "good"
    assert e_low["severity"]  == "good"
    assert "Pitch high" not in e_high["messages"]
    assert "Pitch low"  not in e_low["messages"]


# ── Late warnings ─────────────────────────────────────────────────────────────

def test_a_little_late():
    e = _event(timing=0.07, pitch=0.0)
    assert e["severity"] == "warn"
    assert "A little late" in e["messages"]
    assert "A little early" not in e["messages"]


def test_timing_at_warn_boundary_late():
    # |0.12| <= 0.12 → still warn
    e = _event(timing=0.12, pitch=0.0)
    assert e["severity"] == "warn"
    assert "A little late" in e["messages"]


def test_very_late():
    e = _event(timing=0.13, pitch=0.0)
    assert e["severity"] == "miss"
    assert "Very late" in e["messages"]


def test_very_late_large():
    e = _event(timing=0.5, pitch=0.0)
    assert e["severity"] == "miss"
    assert "Very late" in e["messages"]


# ── Early warnings ────────────────────────────────────────────────────────────

def test_a_little_early():
    e = _event(timing=-0.07, pitch=0.0)
    assert e["severity"] == "warn"
    assert "A little early" in e["messages"]
    assert "A little late" not in e["messages"]


def test_timing_at_warn_boundary_early():
    e = _event(timing=-0.12, pitch=0.0)
    assert e["severity"] == "warn"
    assert "A little early" in e["messages"]


def test_very_early():
    e = _event(timing=-0.13, pitch=0.0)
    assert e["severity"] == "miss"
    assert "Very early" in e["messages"]


# ── Pitch high warnings ───────────────────────────────────────────────────────

def test_pitch_high_warn():
    e = _event(timing=0.0, pitch=35.0)
    assert e["severity"] == "warn"
    assert "Pitch high" in e["messages"]


def test_pitch_at_warn_boundary_high():
    # |50| <= 50 → warn, not miss
    e = _event(timing=0.0, pitch=50.0)
    assert e["severity"] == "warn"
    assert "Pitch high" in e["messages"]


def test_pitch_high_miss():
    e = _event(timing=0.0, pitch=51.0)
    assert e["severity"] == "miss"
    assert "Pitch high" in e["messages"]


def test_pitch_high_miss_large():
    e = _event(timing=0.0, pitch=200.0)
    assert e["severity"] == "miss"
    assert "Pitch high" in e["messages"]


# ── Pitch low warnings ────────────────────────────────────────────────────────

def test_pitch_low_warn():
    e = _event(timing=0.0, pitch=-35.0)
    assert e["severity"] == "warn"
    assert "Pitch low" in e["messages"]


def test_pitch_at_warn_boundary_low():
    e = _event(timing=0.0, pitch=-50.0)
    assert e["severity"] == "warn"
    assert "Pitch low" in e["messages"]


def test_pitch_low_miss():
    e = _event(timing=0.0, pitch=-51.0)
    assert e["severity"] == "miss"
    assert "Pitch low" in e["messages"]


# ── Combined severity (worst dimension wins) ──────────────────────────────────

def test_good_timing_warn_pitch_yields_warn():
    e = _event(timing=0.02, pitch=35.0)
    assert e["severity"] == "warn"
    assert "Good timing" in e["messages"]
    assert "Pitch high"  in e["messages"]


def test_good_timing_miss_pitch_yields_miss():
    e = _event(timing=0.02, pitch=60.0)
    assert e["severity"] == "miss"
    assert "Good timing" in e["messages"]
    assert "Pitch high"  in e["messages"]


def test_warn_timing_miss_pitch_yields_miss():
    e = _event(timing=0.07, pitch=60.0)
    assert e["severity"] == "miss"
    assert "A little late" in e["messages"]
    assert "Pitch high"    in e["messages"]


def test_miss_timing_warn_pitch_yields_miss():
    e = _event(timing=0.15, pitch=-35.0)
    assert e["severity"] == "miss"
    assert "Very late"  in e["messages"]
    assert "Pitch low"  in e["messages"]


# ── No note detected ──────────────────────────────────────────────────────────

def test_no_detected_note_is_miss():
    e = feedback_event(0, _TARGET,
                       {"detected_note": None, "timing_error_s": None,
                        "pitch_error_cents": None})
    assert e["severity"] == "miss"
    assert "No note detected" in e["messages"]


def test_absent_detected_note_key_is_miss():
    e = feedback_event(0, _TARGET, {})
    assert e["severity"] == "miss"
    assert "No note detected" in e["messages"]


def test_no_detected_note_timing_and_pitch_are_none():
    e = feedback_event(0, _TARGET, {"detected_note": None})
    assert e["detected_note"]    is None
    assert e["timing_error_s"]   is None
    assert e["pitch_error_cents"] is None


def test_no_detected_note_no_timing_message():
    e = feedback_event(0, _TARGET, {"detected_note": None})
    assert "Good timing"   not in e["messages"]
    assert "A little late" not in e["messages"]
    assert "Very late"     not in e["messages"]


# ── Low confidence ────────────────────────────────────────────────────────────

def test_low_confidence_bumps_good_to_warn():
    e = _event(timing=0.02, pitch=10.0, confidence=0.3)
    assert e["severity"] == "warn"


def test_low_confidence_does_not_lower_miss():
    e = _event(timing=0.15, pitch=0.0, confidence=0.3)
    assert e["severity"] == "miss"


def test_low_confidence_does_not_lower_warn():
    e = _event(timing=0.07, pitch=0.0, confidence=0.3)
    assert e["severity"] == "warn"


def test_confidence_exactly_at_threshold_is_not_low():
    # confidence == 0.5 is NOT below 0.5 — no penalty.
    e = _event(timing=0.02, pitch=10.0, confidence=0.5)
    assert e["severity"] == "good"


def test_confidence_just_below_threshold_triggers_warn():
    e = _event(timing=0.02, pitch=10.0, confidence=0.499)
    assert e["severity"] == "warn"


def test_none_confidence_no_penalty():
    e = _event(timing=0.02, pitch=10.0, confidence=None)
    assert e["severity"] == "good"


def test_absent_confidence_key_no_penalty():
    e = _event(timing=0.02, pitch=10.0)   # confidence not in result
    assert e["severity"] == "good"
    assert e["confidence"] is None


# ── Missing optional result fields ─────────────────────────────────────────────

def test_absent_timing_no_timing_message():
    e = feedback_event(0, _TARGET, {"detected_note": "D2", "pitch_error_cents": 10.0})
    assert "Good timing"   not in e["messages"]
    assert "A little late" not in e["messages"]
    assert e["timing_error_s"] is None


def test_absent_pitch_no_pitch_message():
    e = feedback_event(0, _TARGET, {"detected_note": "D2", "timing_error_s": 0.02})
    assert "Pitch high" not in e["messages"]
    assert "Pitch low"  not in e["messages"]
    assert e["pitch_error_cents"] is None
    assert e["severity"] == "good"


# ── summarize_feedback ────────────────────────────────────────────────────────

def test_summary_empty_list():
    s = summarize_feedback([])
    assert s["total"]                      == 0
    assert s["good_count"]                 == 0
    assert s["warn_count"]                 == 0
    assert s["miss_count"]                 == 0
    assert s["mean_abs_timing_error_s"]    == 0.0
    assert s["mean_abs_pitch_error_cents"] == 0.0


def test_summary_all_good():
    events = [_event(timing=0.02, pitch=5.0), _event(timing=-0.01, pitch=-3.0)]
    s = summarize_feedback(events)
    assert s["total"]      == 2
    assert s["good_count"] == 2
    assert s["warn_count"] == 0
    assert s["miss_count"] == 0


def test_summary_mixed_severities():
    events = [
        _event(timing=0.02, pitch=5.0),   # good
        _event(timing=0.07, pitch=0.0),   # warn
        _event(timing=0.15, pitch=0.0),   # miss
    ]
    s = summarize_feedback(events)
    assert s["total"]      == 3
    assert s["good_count"] == 1
    assert s["warn_count"] == 1
    assert s["miss_count"] == 1


def test_summary_mean_abs_timing():
    events = [
        _event(timing=0.10,  pitch=0.0),
        _event(timing=-0.06, pitch=0.0),
    ]
    s = summarize_feedback(events)
    assert abs(s["mean_abs_timing_error_s"] - 0.08) < 1e-9


def test_summary_mean_abs_pitch():
    events = [
        _event(timing=0.0, pitch=30.0),
        _event(timing=0.0, pitch=-10.0),
    ]
    s = summarize_feedback(events)
    assert abs(s["mean_abs_pitch_error_cents"] - 20.0) < 1e-9


def test_summary_excludes_none_timing_from_mean():
    miss_ev = feedback_event(0, _TARGET, {"detected_note": None})
    good_ev = _event(timing=0.04, pitch=0.0)
    s = summarize_feedback([miss_ev, good_ev])
    assert abs(s["mean_abs_timing_error_s"] - 0.04) < 1e-9


def test_summary_excludes_none_pitch_from_mean():
    miss_ev = feedback_event(0, _TARGET, {"detected_note": None})
    good_ev = _event(timing=0.0, pitch=20.0)
    s = summarize_feedback([miss_ev, good_ev])
    assert abs(s["mean_abs_pitch_error_cents"] - 20.0) < 1e-9


def test_summary_counts_all_miss_when_no_detections():
    events = [
        feedback_event(i, _TARGET, {"detected_note": None})
        for i in range(4)
    ]
    s = summarize_feedback(events)
    assert s["total"]      == 4
    assert s["miss_count"] == 4
    assert s["good_count"] == 0
    assert s["mean_abs_timing_error_s"]    == 0.0
    assert s["mean_abs_pitch_error_cents"] == 0.0
