"""Tests for core/target_windows.py — no audio hardware."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.target_windows import (
    target_audio_time_s,
    target_gap_s,
    target_analysis_window_samples,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Quarter notes at beats 0, 1, 2, 3.
_TARGETS = [
    {"time": 0},
    {"time": 1},
    {"time": 2},
    {"time": 3},
]

# BPM=120 → beat_s=0.5, count_in=4 → count_in_s=2.0, SR=48000
# gap=0.5 s, duration=min(0.35, 0.6×0.5)=0.30 s
#
#   idx=0: start=2.02 s → 96960,  end=2.32 s → 111360
#   idx=1: start=2.52 s → 120960, end=2.82 s → 135360
#   idx=2: start=3.02 s → 144960, end=3.32 s → 159360
#   idx=3: start=3.52 s → 168960, end=3.82 s → 183360

_BPM      = 120.0
_COUNT_IN = 4
_SR       = 48000


def _window(idx, bpm=_BPM, count_in=_COUNT_IN, sr=_SR, targets=None, **kwargs):
    return target_analysis_window_samples(
        targets if targets is not None else _TARGETS,
        idx, bpm, count_in, sr, **kwargs,
    )


# ── target_audio_time_s ───────────────────────────────────────────────────────

def test_audio_time_first_target():
    # time=0 → only count_in_s = 4 × 0.5 = 2.0
    assert abs(target_audio_time_s(_TARGETS, 0, _BPM, _COUNT_IN) - 2.0) < 1e-9


def test_audio_time_second_target():
    assert abs(target_audio_time_s(_TARGETS, 1, _BPM, _COUNT_IN) - 2.5) < 1e-9


def test_audio_time_last_target():
    assert abs(target_audio_time_s(_TARGETS, 3, _BPM, _COUNT_IN) - 3.5) < 1e-9


def test_audio_time_no_count_in():
    assert abs(target_audio_time_s(_TARGETS, 0, _BPM, count_in_beats=0) - 0.0) < 1e-9


def test_audio_time_scales_with_bpm():
    # BPM=60: beat_s=1.0; count_in_s=4.0; idx=1 → 4.0 + 1.0 = 5.0
    assert abs(target_audio_time_s(_TARGETS, 1, 60.0, _COUNT_IN) - 5.0) < 1e-9


# ── target_gap_s ──────────────────────────────────────────────────────────────

def test_gap_first_target():
    # (1 - 0) × 0.5 = 0.5 s
    assert abs(target_gap_s(_TARGETS, 0, _BPM) - 0.5) < 1e-9


def test_gap_middle_target():
    # (2 - 1) × 0.5 = 0.5 s
    assert abs(target_gap_s(_TARGETS, 1, _BPM) - 0.5) < 1e-9


def test_gap_last_target_reuses_previous():
    # idx=3 is last; reuses (3 - 2) × 0.5 = 0.5 s
    assert abs(target_gap_s(_TARGETS, 3, _BPM) - 0.5) < 1e-9


def test_gap_last_equals_second_to_last():
    assert abs(target_gap_s(_TARGETS, 3, _BPM) - target_gap_s(_TARGETS, 2, _BPM)) < 1e-9


def test_gap_single_target_default():
    assert abs(target_gap_s([{"time": 0}], 0, _BPM) - 0.5) < 1e-9


def test_gap_scales_with_bpm():
    # BPM=60: beat_s=1.0; gap between beats 0 and 1 = 1.0 s
    assert abs(target_gap_s(_TARGETS, 0, 60.0) - 1.0) < 1e-9


def test_gap_zero_raises():
    bad = [{"time": 0}, {"time": 0}]
    with pytest.raises(ValueError):
        target_gap_s(bad, 0, _BPM)


def test_gap_negative_raises():
    bad = [{"time": 1}, {"time": 0}]   # reversed — gap < 0
    with pytest.raises(ValueError):
        target_gap_s(bad, 0, _BPM)


def test_gap_last_target_negative_reuse_raises():
    bad = [{"time": 2}, {"time": 1}]
    with pytest.raises(ValueError):
        target_gap_s(bad, 1, _BPM)


# ── target_analysis_window_samples — first target window ─────────────────────

def test_first_target_window():
    start, end = _window(0)
    assert start == 96960
    assert end   == 111360


def test_first_target_window_start_nonzero():
    start, _ = _window(0)
    assert start > 0


# ── Middle target window ──────────────────────────────────────────────────────

def test_middle_target_window():
    start, end = _window(1)
    assert start == 120960
    assert end   == 135360


def test_middle_window_later_than_first():
    start0, _ = _window(0)
    start1, _ = _window(1)
    assert start1 > start0


# ── Last target reuses previous gap ──────────────────────────────────────────

def test_last_target_window():
    start, end = _window(3)
    assert start == 168960
    assert end   == 183360


def test_last_target_duration_equals_penultimate():
    start2, end2 = _window(2)
    start3, end3 = _window(3)
    assert (end2 - start2) == (end3 - start3)


# ── Single target default gap ─────────────────────────────────────────────────

def test_single_target_window():
    # gap defaults to 0.5 s → same duration as standard quarter-note case
    single = [{"time": 0}]
    start, end = target_analysis_window_samples(single, 0, _BPM, _COUNT_IN, _SR)
    assert start == 96960   # 2.02 × 48000
    assert end   == 111360  # 2.32 × 48000


def test_single_target_duration():
    single = [{"time": 0}]
    start, end = target_analysis_window_samples(single, 0, _BPM, _COUNT_IN, _SR)
    dur = (end - start) / _SR
    assert abs(dur - 0.30) < 1e-6


# ── Count-in shifts window ────────────────────────────────────────────────────

def test_count_in_zero_shifts_window_earlier():
    # count_in=0: start=0.02 s → 960, end=0.32 s → 15360
    start, end = _window(0, count_in=0)
    assert start == 960
    assert end   == 15360


def test_count_in_four_vs_zero_difference():
    # 4 beats × 0.5 s/beat × 48000 samples/s = 96000 samples
    start4, _ = _window(0, count_in=4)
    start0, _ = _window(0, count_in=0)
    assert start4 - start0 == 96000


# ── BPM changes window ────────────────────────────────────────────────────────

def test_bpm_60_window():
    # BPM=60: beat_s=1.0, count_in_s=4.0; gap=1.0 → capped at max_window_s=0.35
    # start=4.02 → 192960; end=4.37 → 209760
    start, end = _window(0, bpm=60.0)
    assert start == 192960
    assert end   == 209760


def test_bpm_120_duration_uses_gap_fraction():
    # gap=0.5, gap_fraction×gap=0.3 < max_window_s=0.35 → duration=0.3
    start, end = _window(0)
    dur = (end - start) / _SR
    assert abs(dur - 0.30) < 1e-6


def test_bpm_60_duration_uses_max_window_cap():
    # gap=1.0, gap_fraction×gap=0.6 > max_window_s=0.35 → capped at 0.35
    start, end = _window(0, bpm=60.0)
    dur = (end - start) / _SR
    assert abs(dur - 0.35) < 1e-6


# ── Sample rate changes sample indices ────────────────────────────────────────

def test_sample_rate_44100():
    # start=2.02 × 44100=89082; end=2.32 × 44100=102312
    start, end = _window(0, sr=44100)
    assert start == 89082
    assert end   == 102312


def test_sample_rate_proportional_scaling():
    start48, end48 = _window(0, sr=48000)
    start44, end44 = _window(0, sr=44100)
    ratio = 44100 / 48000
    assert abs(start44 / start48 - ratio) < 1e-4
    assert abs(end44   / end48   - ratio) < 1e-4


# ── round() — indices are Python ints ────────────────────────────────────────

def test_sample_indices_are_python_int():
    # round() returns a Python int; numpy equivalents return float or int64.
    # int() would truncate a value like 2.9999... to 2 instead of 3.
    start, end = _window(0)
    assert type(start) is int
    assert type(end)   is int


def test_return_is_tuple_of_two():
    result = _window(0)
    assert isinstance(result, tuple)
    assert len(result) == 2


# ── gap_fraction vs max_window_s selection ────────────────────────────────────

def test_gap_fraction_applied_when_gap_small():
    # Dense targets: gap = 0.25 beats × 0.5 s/beat = 0.125 s
    # gap_fraction × 0.125 = 0.075 < 0.35 → duration = 0.075
    # start=2.02 → 96960; end=2.095 → 100560
    dense = [{"time": i * 0.25} for i in range(4)]
    start, end = target_analysis_window_samples(dense, 0, _BPM, _COUNT_IN, _SR)
    assert start == 96960
    assert end   == 100560


def test_max_window_applied_when_gap_large():
    # Sparse targets: gap = 2 beats × 1.0 s/beat = 2.0 s at BPM=60
    # gap_fraction × 2.0 = 1.2 > 0.35 → capped; start=4.02 → 192960
    sparse = [{"time": 0}, {"time": 2}]
    start, end = target_analysis_window_samples(sparse, 0, 60.0, _COUNT_IN, _SR)
    assert start == 192960
    assert end   == 209760


# ── Validation — invalid idx ──────────────────────────────────────────────────

def test_invalid_idx_too_large_raises():
    with pytest.raises(ValueError, match="out of range"):
        _window(4)   # valid range: [0, 3]


def test_invalid_idx_negative_raises():
    with pytest.raises(ValueError, match="out of range"):
        _window(-1)


def test_invalid_idx_empty_targets_raises():
    with pytest.raises(ValueError):
        target_analysis_window_samples([], 0, _BPM, _COUNT_IN, _SR)


# ── Validation — nonpositive bpm ──────────────────────────────────────────────

def test_bpm_zero_raises():
    with pytest.raises(ValueError, match="bpm"):
        _window(0, bpm=0.0)


def test_bpm_negative_raises():
    with pytest.raises(ValueError, match="bpm"):
        _window(0, bpm=-120.0)


# ── Validation — nonpositive sample_rate ──────────────────────────────────────

def test_sample_rate_zero_raises():
    with pytest.raises(ValueError, match="sample_rate"):
        _window(0, sr=0)


def test_sample_rate_negative_raises():
    with pytest.raises(ValueError, match="sample_rate"):
        _window(0, sr=-48000)


# ── Validation — zero or negative target gap ──────────────────────────────────

def test_zero_gap_targets_raises():
    bad = [{"time": 0}, {"time": 0}]
    with pytest.raises(ValueError):
        target_analysis_window_samples(bad, 0, _BPM, _COUNT_IN, _SR)


def test_negative_gap_targets_raises():
    bad = [{"time": 1}, {"time": 0}]
    with pytest.raises(ValueError):
        target_analysis_window_samples(bad, 0, _BPM, _COUNT_IN, _SR)
