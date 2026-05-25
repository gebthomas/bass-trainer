"""Tests for the WAV recording helper _float_to_int16 in live_feedback_demo.py.

sounddevice is mocked so no audio hardware is required.

Test matrix
-----------
Output dtype and shape
    1.  Output dtype is int16.
    2.  Output shape matches input shape.
    3.  Empty array → empty int16 array.

Value mapping
    4.  0.0 → 0.
    5.  1.0 → 32767 (max).
    6.  -1.0 → -32767 (min, symmetric with positive).
    7.  0.5 → round(0.5 × 32767).
    8.  -0.5 → round(-0.5 × 32767).
    9.  All output values in [-32767, 32767].

Clipping
    10. Values above 1.0 are clipped to 32767.
    11. Values below -1.0 are clipped to -32767.
    12. Far-out-of-range values do not produce wrap-around.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_sd_mock = unittest.mock.MagicMock()
_sd_mock.query_devices.return_value = []

with unittest.mock.patch.dict("sys.modules", {"sounddevice": _sd_mock}):
    import importlib
    import scripts.live_feedback_demo as _demo_mod
    importlib.reload(_demo_mod)

_float_to_int16 = _demo_mod._float_to_int16


class TestFloatToInt16:

    # ── dtype and shape ───────────────────────────────────────────────────────

    def test_output_dtype_is_int16(self):
        assert _float_to_int16(np.zeros(10)).dtype == np.int16

    def test_output_shape_preserved(self):
        audio = np.linspace(-1.0, 1.0, 500)
        assert _float_to_int16(audio).shape == audio.shape

    def test_empty_array_returns_empty_int16(self):
        result = _float_to_int16(np.array([]))
        assert result.dtype == np.int16
        assert result.size == 0

    # ── value mapping ─────────────────────────────────────────────────────────

    def test_zero_maps_to_zero(self):
        assert _float_to_int16(np.array([0.0]))[0] == 0

    def test_positive_one_maps_to_max(self):
        assert _float_to_int16(np.array([1.0]))[0] == 32767

    def test_negative_one_maps_to_min(self):
        assert _float_to_int16(np.array([-1.0]))[0] == -32767

    def test_half_amplitude_positive(self):
        result  = _float_to_int16(np.array([0.5]))[0]
        expected = int(0.5 * 32767)
        assert result == expected

    def test_half_amplitude_negative(self):
        result   = _float_to_int16(np.array([-0.5]))[0]
        expected = int(-0.5 * 32767)
        assert result == expected

    def test_all_outputs_in_valid_int16_range(self):
        audio  = np.random.default_rng(42).uniform(-1.0, 1.0, 10_000)
        result = _float_to_int16(audio)
        assert np.all(result >= -32767)
        assert np.all(result <= 32767)

    # ── clipping ──────────────────────────────────────────────────────────────

    def test_values_above_one_clipped_to_max(self):
        result = _float_to_int16(np.array([1.5, 2.0, 100.0]))
        assert np.all(result == 32767)

    def test_values_below_minus_one_clipped_to_min(self):
        result = _float_to_int16(np.array([-1.5, -2.0, -100.0]))
        assert np.all(result == -32767)

    def test_clipping_does_not_wrap_around(self):
        # int16 wraps at ±32768; clipping must prevent that.
        result = _float_to_int16(np.array([999.0]))
        assert result[0] == 32767
        result = _float_to_int16(np.array([-999.0]))
        assert result[0] == -32767

    def test_just_at_clip_boundary(self):
        assert _float_to_int16(np.array([1.0]))[0]  == 32767
        assert _float_to_int16(np.array([-1.0]))[0] == -32767
