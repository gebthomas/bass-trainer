"""Tests for core/timing_policy.py — canonical match-window formula.

All tests are pure Python; no audio hardware required.

Test matrix
-----------
Nominal outputs
  1.  120 bpm → 0.25 s  (half a beat, within default clamp range).
  2.   90 bpm → 0.333 s (within default clamp range).

Upper clamp
  3.   60 bpm → 0.35 s  (unclamped = 0.50 s, clamped to DEFAULT_MAX).
  4.   40 bpm → 0.35 s  (deeply clamped).

Lower clamp
  5.  400 bpm → 0.10 s  (unclamped = 0.075 s, clamped to DEFAULT_MIN).
  6.  300 bpm → 0.10 s  (unclamped = 0.10 s, exactly at boundary).

Boundary — no clamp applied
  7.  At exact lower boundary bpm (300): result == DEFAULT_MIN_MATCH_WINDOW_S.
  8.  At exact upper boundary bpm (≈85.7): result == DEFAULT_MAX_MATCH_WINDOW_S.

Invalid bpm
  9.  bpm == 0 raises ValueError.
  10. bpm < 0 raises ValueError.

Invalid min_s / max_s
  11. min_s == 0 raises ValueError.
  12. min_s < 0 raises ValueError.
  13. max_s == 0 raises ValueError.
  14. max_s < 0 raises ValueError.
  15. min_s > max_s raises ValueError.

Custom clamp range
  16. Custom min_s / max_s respected when bpm falls outside default range.
  17. Custom min_s == max_s (degenerate clamp) always returns that value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.timing_policy import (
    DEFAULT_MAX_MATCH_WINDOW_S,
    DEFAULT_MIN_MATCH_WINDOW_S,
    match_window_s,
)


# ── 1–2: Nominal outputs ──────────────────────────────────────────────────────

def test_120_bpm():
    assert match_window_s(120.0) == pytest.approx(0.25)


def test_90_bpm():
    assert match_window_s(90.0) == pytest.approx(30.0 / 90.0)


# ── 3–4: Upper clamp ─────────────────────────────────────────────────────────

def test_60_bpm_clamps_to_max():
    result = match_window_s(60.0)
    assert result == pytest.approx(DEFAULT_MAX_MATCH_WINDOW_S)


def test_40_bpm_clamps_to_max():
    result = match_window_s(40.0)
    assert result == pytest.approx(DEFAULT_MAX_MATCH_WINDOW_S)


# ── 5–6: Lower clamp ─────────────────────────────────────────────────────────

def test_400_bpm_clamps_to_min():
    result = match_window_s(400.0)
    assert result == pytest.approx(DEFAULT_MIN_MATCH_WINDOW_S)


def test_300_bpm_at_lower_boundary():
    # 30.0 / 300.0 = 0.10 exactly — sits on the clamp boundary
    result = match_window_s(300.0)
    assert result == pytest.approx(DEFAULT_MIN_MATCH_WINDOW_S)


# ── 7–8: Exact boundary values ───────────────────────────────────────────────

def test_at_lower_boundary_bpm():
    bpm_at_min = 30.0 / DEFAULT_MIN_MATCH_WINDOW_S  # 300 bpm
    assert match_window_s(bpm_at_min) == pytest.approx(DEFAULT_MIN_MATCH_WINDOW_S)


def test_at_upper_boundary_bpm():
    bpm_at_max = 30.0 / DEFAULT_MAX_MATCH_WINDOW_S  # ≈85.7 bpm
    assert match_window_s(bpm_at_max) == pytest.approx(DEFAULT_MAX_MATCH_WINDOW_S)


# ── 9–10: Invalid bpm ────────────────────────────────────────────────────────

def test_zero_bpm_raises():
    with pytest.raises(ValueError, match="bpm"):
        match_window_s(0.0)


def test_negative_bpm_raises():
    with pytest.raises(ValueError, match="bpm"):
        match_window_s(-120.0)


# ── 11–15: Invalid min_s / max_s ─────────────────────────────────────────────

def test_zero_min_s_raises():
    with pytest.raises(ValueError, match="min_s"):
        match_window_s(120.0, min_s=0.0, max_s=0.35)


def test_negative_min_s_raises():
    with pytest.raises(ValueError, match="min_s"):
        match_window_s(120.0, min_s=-0.1, max_s=0.35)


def test_zero_max_s_raises():
    with pytest.raises(ValueError, match="max_s"):
        match_window_s(120.0, min_s=0.10, max_s=0.0)


def test_negative_max_s_raises():
    with pytest.raises(ValueError, match="max_s"):
        match_window_s(120.0, min_s=0.10, max_s=-0.1)


def test_min_s_greater_than_max_s_raises():
    with pytest.raises(ValueError, match="min_s"):
        match_window_s(120.0, min_s=0.40, max_s=0.20)


# ── 16–17: Custom clamp range ────────────────────────────────────────────────

def test_custom_clamp_range():
    # At 60 bpm, unclamped = 0.50 s.
    # With custom max_s=0.45 s, result should be 0.45 s.
    result = match_window_s(60.0, min_s=0.05, max_s=0.45)
    assert result == pytest.approx(0.45)


def test_degenerate_clamp_min_equals_max():
    # min_s == max_s: always return that single value regardless of bpm.
    result = match_window_s(120.0, min_s=0.20, max_s=0.20)
    assert result == pytest.approx(0.20)

    result = match_window_s(60.0, min_s=0.20, max_s=0.20)
    assert result == pytest.approx(0.20)
