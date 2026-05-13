"""Tests for core/ring_buffer.py — no audio hardware."""

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ring_buffer import RingBuffer


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_buf(capacity=10, channels=1):
    """Return a RingBuffer with exactly *capacity* samples of storage."""
    return RingBuffer(sample_rate=capacity, max_seconds=1.0, channels=channels)


def _ramp(n, start=0):
    """1-D array [start, start+1, ..., start+n-1] as float64."""
    return np.arange(start, start + n, dtype=np.float64)


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_current_sample_is_zero():
    buf = make_buf()
    assert buf.current_sample() == 0


def test_initial_available_range():
    buf = make_buf()
    assert buf.available_range() == (0, 0)


def test_initial_buffer_zeros():
    buf = make_buf(capacity=4)
    buf.add(np.zeros(4))
    npt.assert_array_equal(buf.get_window(0, 4), np.zeros(4))


# ── add() — mono ─────────────────────────────────────────────────────────────

def test_add_mono_advances_current_sample():
    buf = make_buf()
    buf.add(_ramp(5))
    assert buf.current_sample() == 5


def test_add_mono_available_range_partial():
    buf = make_buf(capacity=10)
    buf.add(_ramp(6))
    assert buf.available_range() == (0, 6)


def test_add_mono_available_range_full():
    buf = make_buf(capacity=10)
    buf.add(_ramp(10))
    assert buf.available_range() == (0, 10)


def test_add_mono_available_range_overflow():
    buf = make_buf(capacity=10)
    buf.add(_ramp(15))
    assert buf.available_range() == (5, 15)


def test_add_multiple_chunks_current_sample():
    buf = make_buf(capacity=10)
    buf.add(_ramp(4))
    buf.add(_ramp(4, start=4))
    assert buf.current_sample() == 8


def test_add_multiple_chunks_available_range():
    buf = make_buf(capacity=10)
    buf.add(_ramp(6))
    buf.add(_ramp(6, start=6))
    assert buf.available_range() == (2, 12)


def test_add_empty_chunk_noop():
    buf = make_buf(capacity=10)
    buf.add(_ramp(5))
    buf.add(np.array([]))
    assert buf.current_sample() == 5


# ── add() — 2-D mono (n, 1) ──────────────────────────────────────────────────

def test_add_2d_mono_shape():
    buf = make_buf(capacity=6)
    buf.add(_ramp(6).reshape(-1, 1))
    assert buf.current_sample() == 6


# ── get_window() — simple (no wraparound) ────────────────────────────────────

def test_get_window_simple():
    buf = make_buf(capacity=10)
    buf.add(_ramp(8))
    npt.assert_array_equal(buf.get_window(2, 6), _ramp(4, start=2))


def test_get_window_full_buffer():
    buf = make_buf(capacity=10)
    buf.add(_ramp(10))
    npt.assert_array_equal(buf.get_window(0, 10), _ramp(10))


def test_get_window_mono_returns_1d():
    buf = make_buf(capacity=8)
    buf.add(_ramp(8))
    result = buf.get_window(0, 8)
    assert result.ndim == 1


def test_get_window_single_sample():
    buf = make_buf(capacity=8)
    buf.add(_ramp(8))
    npt.assert_array_equal(buf.get_window(3, 4), np.array([3.0]))


# ── get_window() — partial fill (no wraparound yet) ──────────────────────────

def test_get_window_partial_fill_from_start():
    buf = make_buf(capacity=10)
    buf.add(_ramp(6))
    npt.assert_array_equal(buf.get_window(0, 6), _ramp(6))


def test_get_window_partial_fill_subset():
    buf = make_buf(capacity=10)
    buf.add(_ramp(6))
    npt.assert_array_equal(buf.get_window(2, 5), _ramp(3, start=2))


# ── get_window() — after wraparound ──────────────────────────────────────────

def test_get_window_after_wraparound():
    # capacity=10; write 15 samples → available [5, 15)
    buf = make_buf(capacity=10)
    buf.add(_ramp(15))
    npt.assert_array_equal(buf.get_window(7, 13), _ramp(6, start=7))


def test_get_window_full_wraparound():
    buf = make_buf(capacity=10)
    buf.add(_ramp(15))
    npt.assert_array_equal(buf.get_window(5, 15), _ramp(10, start=5))


def test_get_window_crosses_physical_boundary():
    # capacity=10; write 15 samples; [8,13) crosses end of physical buffer
    buf = make_buf(capacity=10)
    buf.add(_ramp(15))
    npt.assert_array_equal(buf.get_window(8, 13), _ramp(5, start=8))


def test_get_window_multiple_wraps():
    # Write 3× capacity to ensure the buffer has wrapped multiple times.
    buf = make_buf(capacity=10)
    buf.add(_ramp(30))
    # available [20, 30)
    npt.assert_array_equal(buf.get_window(22, 28), _ramp(6, start=22))


# ── get_window() — reject too-old ────────────────────────────────────────────

def test_get_window_rejects_too_old():
    buf = make_buf(capacity=10)
    buf.add(_ramp(15))
    with pytest.raises(ValueError, match="too old"):
        buf.get_window(4, 10)


def test_get_window_rejects_start_before_oldest():
    buf = make_buf(capacity=10)
    buf.add(_ramp(15))
    with pytest.raises(ValueError):
        buf.get_window(0, 10)


# ── get_window() — reject future ─────────────────────────────────────────────

def test_get_window_rejects_future_end():
    buf = make_buf(capacity=10)
    buf.add(_ramp(8))
    with pytest.raises(ValueError, match="beyond current_sample"):
        buf.get_window(2, 9)


def test_get_window_rejects_fully_future():
    buf = make_buf(capacity=10)
    with pytest.raises(ValueError):
        buf.get_window(5, 10)


# ── get_window() — reject invalid range ──────────────────────────────────────

def test_get_window_rejects_empty_range():
    buf = make_buf(capacity=10)
    buf.add(_ramp(8))
    with pytest.raises(ValueError):
        buf.get_window(4, 4)


def test_get_window_rejects_reversed_range():
    buf = make_buf(capacity=10)
    buf.add(_ramp(8))
    with pytest.raises(ValueError):
        buf.get_window(6, 3)


# ── Stereo / multi-channel ────────────────────────────────────────────────────

def test_stereo_add_and_retrieve():
    buf = make_buf(capacity=8, channels=2)
    chunk = np.column_stack([_ramp(8), _ramp(8, start=100)])
    buf.add(chunk)
    result = buf.get_window(0, 8)
    assert result.shape == (8, 2)
    npt.assert_array_equal(result[:, 0], _ramp(8))
    npt.assert_array_equal(result[:, 1], _ramp(8, start=100))


def test_stereo_get_window_returns_2d():
    buf = make_buf(capacity=6, channels=2)
    buf.add(np.ones((6, 2)))
    result = buf.get_window(0, 6)
    assert result.ndim == 2
    assert result.shape[1] == 2


def test_stereo_wraparound_preserves_values():
    buf = make_buf(capacity=8, channels=2)
    data = np.column_stack([_ramp(12), _ramp(12, start=100)])
    buf.add(data)
    # available [4, 12)
    result = buf.get_window(6, 10)
    npt.assert_array_equal(result[:, 0], _ramp(4, start=6))
    npt.assert_array_equal(result[:, 1], _ramp(4, start=106))


def test_three_channel_add_and_retrieve():
    buf = make_buf(capacity=6, channels=3)
    chunk = np.ones((6, 3)) * np.array([1.0, 2.0, 3.0])
    buf.add(chunk)
    result = buf.get_window(0, 6)
    assert result.shape == (6, 3)
    npt.assert_array_equal(result[:, 2], np.full(6, 3.0))


# ── Wrong channel count raises ────────────────────────────────────────────────

def test_wrong_channel_count_raises():
    buf = make_buf(capacity=8, channels=2)
    with pytest.raises(ValueError, match="channel"):
        buf.add(_ramp(4))   # 1-D into 2-channel buffer


def test_wrong_2d_channel_count_raises():
    buf = make_buf(capacity=8, channels=2)
    with pytest.raises(ValueError, match="channel"):
        buf.add(np.ones((4, 3)))


def test_mono_buffer_rejects_2d_multichannel():
    buf = make_buf(capacity=8, channels=1)
    with pytest.raises(ValueError):
        buf.add(np.ones((4, 2)))


# ── current_sample advances correctly ────────────────────────────────────────

def test_current_sample_accumulates_across_adds():
    buf = make_buf(capacity=20)
    buf.add(_ramp(3))
    buf.add(_ramp(7))
    buf.add(_ramp(5))
    assert buf.current_sample() == 15


def test_current_sample_advances_past_capacity():
    buf = make_buf(capacity=10)
    buf.add(_ramp(25))
    assert buf.current_sample() == 25


# ── available_range updates correctly after overflow ─────────────────────────

def test_available_range_before_full():
    buf = make_buf(capacity=10)
    buf.add(_ramp(7))
    assert buf.available_range() == (0, 7)


def test_available_range_exactly_full():
    buf = make_buf(capacity=10)
    buf.add(_ramp(10))
    assert buf.available_range() == (0, 10)


def test_available_range_one_past_full():
    buf = make_buf(capacity=10)
    buf.add(_ramp(11))
    assert buf.available_range() == (1, 11)


def test_available_range_large_overflow():
    buf = make_buf(capacity=10)
    buf.add(_ramp(37))
    assert buf.available_range() == (27, 37)


def test_available_range_incremental_writes():
    buf = make_buf(capacity=10)
    for i in range(5):
        buf.add(_ramp(4, start=i * 4))
    # 20 samples total; capacity=10 → oldest=10
    assert buf.available_range() == (10, 20)
