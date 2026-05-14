"""Tests for core/feedback_scoring.py — no audio hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.feedback_scoring import score_timing


# ── Helpers ───────────────────────────────────────────────────────────────────

_TOL = 0.050   # default tolerance_s


def _ev(detected=True, onset_found=True, onset_time_s=0.0):
    return {"detected": detected, "onset_found": onset_found,
            "onset_time_s": onset_time_s}


def _score(onset_time_s=0.0, **kw):
    return score_timing(_ev(onset_time_s=onset_time_s), **kw)


# ── Return structure ──────────────────────────────────────────────────────────

def test_return_keys_present():
    out = _score()
    for k in ("status", "offset_s", "offset_ms", "target_index"):
        assert k in out, f"missing key: {k!r}"


def test_status_is_str():
    assert isinstance(_score()["status"], str)


def test_target_index_none_by_default():
    assert _score()["target_index"] is None


# ── Miss ──────────────────────────────────────────────────────────────────────

def test_miss_when_not_detected():
    out = score_timing(_ev(detected=False))
    assert out["status"] == "miss"


def test_miss_when_onset_not_found():
    out = score_timing(_ev(onset_found=False))
    assert out["status"] == "miss"


def test_miss_when_both_false():
    out = score_timing(_ev(detected=False, onset_found=False))
    assert out["status"] == "miss"


def test_miss_offset_s_none():
    assert score_timing(_ev(detected=False))["offset_s"] is None


def test_miss_offset_ms_none():
    assert score_timing(_ev(detected=False))["offset_ms"] is None


def test_miss_target_index_passed_through():
    out = score_timing(_ev(detected=False), target_index=7)
    assert out["target_index"] == 7


# ── Missing onset (onset_found=False, detected=True) ─────────────────────────

def test_missing_onset_status_miss():
    out = score_timing(_ev(onset_found=False, onset_time_s=None))
    assert out["status"] == "miss"


def test_missing_onset_offset_s_none():
    out = score_timing(_ev(onset_found=False, onset_time_s=None))
    assert out["offset_s"] is None


def test_missing_onset_offset_ms_none():
    out = score_timing(_ev(onset_found=False, onset_time_s=None))
    assert out["offset_ms"] is None


# ── Early ─────────────────────────────────────────────────────────────────────

def test_early_when_before_negative_tolerance():
    out = _score(-0.10)
    assert out["status"] == "early"


def test_early_just_past_tolerance():
    out = _score(-(_TOL + 0.001))
    assert out["status"] == "early"


def test_early_offset_s_returned():
    out = _score(-0.10)
    assert abs(out["offset_s"] - (-0.10)) < 1e-9


def test_early_offset_ms():
    out = _score(-0.10)
    assert abs(out["offset_ms"] - (-100.0)) < 1e-9


def test_early_target_index_passed():
    out = score_timing(_ev(onset_time_s=-0.1), target_index=2)
    assert out["target_index"] == 2


# ── Late ──────────────────────────────────────────────────────────────────────

def test_late_when_after_positive_tolerance():
    out = _score(0.10)
    assert out["status"] == "late"


def test_late_just_past_tolerance():
    out = _score(_TOL + 0.001)
    assert out["status"] == "late"


def test_late_offset_s_returned():
    out = _score(0.10)
    assert abs(out["offset_s"] - 0.10) < 1e-9


def test_late_offset_ms():
    out = _score(0.10)
    assert abs(out["offset_ms"] - 100.0) < 1e-9


def test_late_large_offset():
    out = _score(1.0)
    assert out["status"] == "late"


# ── On time ───────────────────────────────────────────────────────────────────

def test_on_time_at_zero():
    assert _score(0.0)["status"] == "on_time"


def test_on_time_small_positive():
    assert _score(0.04)["status"] == "on_time"


def test_on_time_small_negative():
    assert _score(-0.04)["status"] == "on_time"


def test_on_time_at_positive_boundary():
    # Exactly at +tolerance_s is NOT late (strict >).
    assert _score(_TOL)["status"] == "on_time"


def test_on_time_at_negative_boundary():
    # Exactly at -tolerance_s is NOT early (strict <).
    assert _score(-_TOL)["status"] == "on_time"


def test_on_time_offset_s_returned():
    out = _score(0.03)
    assert abs(out["offset_s"] - 0.03) < 1e-9


def test_on_time_offset_ms_is_1000x_offset_s():
    out = _score(0.03)
    assert abs(out["offset_ms"] - 30.0) < 1e-9


def test_on_time_target_index_passed():
    out = score_timing(_ev(onset_time_s=0.02), target_index=0)
    assert out["target_index"] == 0


# ── Custom tolerance ──────────────────────────────────────────────────────────

def test_custom_wider_tolerance_keeps_on_time():
    # 0.08 s onset with 0.10 s tolerance → on_time
    out = score_timing(_ev(onset_time_s=0.08), tolerance_s=0.10)
    assert out["status"] == "on_time"


def test_custom_wider_tolerance_late_beyond_it():
    out = score_timing(_ev(onset_time_s=0.12), tolerance_s=0.10)
    assert out["status"] == "late"


def test_custom_narrower_tolerance_makes_late():
    # 0.04 s onset is on_time at default 0.05 s, but late at 0.02 s
    assert _score(0.04)["status"]                                    == "on_time"
    assert score_timing(_ev(onset_time_s=0.04), tolerance_s=0.02)["status"] == "late"


def test_custom_narrower_tolerance_makes_early():
    assert _score(-0.04)["status"]                                    == "on_time"
    assert score_timing(_ev(onset_time_s=-0.04), tolerance_s=0.02)["status"] == "early"


def test_custom_tolerance_boundary_still_on_time():
    # At exactly the custom tolerance boundary → on_time (strict inequality).
    out = score_timing(_ev(onset_time_s=0.10), tolerance_s=0.10)
    assert out["status"] == "on_time"
