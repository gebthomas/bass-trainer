"""Tests for click_schedule and render_click_track in live_feedback_demo.py.

sounddevice is mocked so no audio hardware is required.

Test matrix
-----------
click_schedule — schedule generation
    1.  count_in=2, n_beats=4 → 6 entries total.
    2.  count_in=2, n_beats=0 → 2 entries (count-in only).
    3.  count_in=0, n_beats=3 → 3 entries (target beats only).
    4.  Both zero → empty list.
    5.  First entry time is 0.0.
    6.  Spacing between consecutive times equals beat_s.
    7.  Entry 0 carries accent_freq.
    8.  Entries 1+ carry beat_freq.
    9.  Custom accent_freq / beat_freq are used.
    10. Times are monotonically increasing.
    11. Last time equals (count_in + n_beats − 1) * beat_s.

render_click_track — audio rendering
    12. Empty schedule → empty float32 array.
    13. dtype is float32.
    14. Single click → non-silent near its scheduled sample.
    15. Two clicks → non-silent near both positions.
    16. Output length covers at least last_time + click_s seconds.
    17. Samples before first click are silent.
    18. Amplitude does not exceed the supplied click_amp.

Argument parsing — --click flag
    19. Default: args.click is False.
    20. --click flag sets args.click to True.
    21. --click and --no-click can coexist in the namespace (no argparse error).
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Import demo module without triggering sounddevice hardware queries ─────────

_sd_mock = unittest.mock.MagicMock()
_sd_mock.query_devices.return_value = []

with unittest.mock.patch.dict("sys.modules", {"sounddevice": _sd_mock}):
    import importlib
    import scripts.live_feedback_demo as _demo_mod
    importlib.reload(_demo_mod)

click_schedule    = _demo_mod.click_schedule
render_click_track = _demo_mod.render_click_track
_parse_args       = _demo_mod._parse_args
ACCENT_FREQ       = _demo_mod.ACCENT_FREQ
BEAT_FREQ         = _demo_mod.BEAT_FREQ
CLICK_S           = _demo_mod.CLICK_S
CLICK_AMP         = _demo_mod.CLICK_AMP


def _parse(*argv: str):
    with unittest.mock.patch("sys.argv", ["live_feedback_demo.py", *argv]):
        return _parse_args()


# ── click_schedule ────────────────────────────────────────────────────────────

class TestClickSchedule:

    def test_total_count_count_in_plus_n_beats(self):
        assert len(click_schedule(60.0, 2, 4)) == 6

    def test_n_beats_zero_gives_count_in_only(self):
        assert len(click_schedule(60.0, 2, 0)) == 2

    def test_count_in_zero_gives_n_beats_only(self):
        assert len(click_schedule(60.0, 0, 3)) == 3

    def test_both_zero_gives_empty(self):
        assert click_schedule(60.0, 0, 0) == []

    def test_first_time_is_zero(self):
        sched = click_schedule(120.0, 2, 4)
        assert sched[0][0] == pytest.approx(0.0)

    def test_spacing_equals_beat_s(self):
        bpm   = 90.0
        beat_s = 60.0 / bpm
        sched  = click_schedule(bpm, 2, 4)
        times  = [t for t, _ in sched]
        diffs  = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        assert all(pytest.approx(d) == beat_s for d in diffs)

    def test_entry_zero_uses_accent_freq(self):
        sched = click_schedule(60.0, 2, 4)
        assert sched[0][1] == pytest.approx(ACCENT_FREQ)

    def test_subsequent_entries_use_beat_freq(self):
        sched = click_schedule(60.0, 2, 4)
        for _, freq in sched[1:]:
            assert freq == pytest.approx(BEAT_FREQ)

    def test_custom_freqs_used(self):
        sched = click_schedule(60.0, 2, 2, accent_freq=1000.0, beat_freq=500.0)
        assert sched[0][1] == pytest.approx(1000.0)
        for _, freq in sched[1:]:
            assert freq == pytest.approx(500.0)

    def test_times_monotonically_increasing(self):
        sched = click_schedule(120.0, 3, 5)
        times = [t for t, _ in sched]
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def test_last_time_equals_expected(self):
        bpm    = 60.0
        beat_s = 60.0 / bpm
        sched  = click_schedule(bpm, 2, 4)
        expected_last = (2 + 4 - 1) * beat_s
        assert sched[-1][0] == pytest.approx(expected_last)


# ── render_click_track ────────────────────────────────────────────────────────

SR = 16_000   # small rate keeps arrays fast


class TestRenderClickTrack:

    def test_empty_schedule_empty_array(self):
        result = render_click_track([], SR)
        assert result.size == 0

    def test_dtype_is_float32(self):
        sched  = click_schedule(60.0, 1, 0)
        result = render_click_track(sched, SR)
        assert result.dtype == np.float32

    def test_single_click_non_silent_at_scheduled_time(self):
        time_s = 0.5
        sched  = [(time_s, 440.0)]
        result = render_click_track(sched, SR, click_s=0.05, click_amp=0.5)
        peak_sample = int(time_s * SR)
        # There should be non-negligible energy near the click onset.
        window = result[peak_sample: peak_sample + int(0.01 * SR)]
        assert np.max(np.abs(window)) > 0.1

    def test_two_clicks_both_non_silent(self):
        sched  = [(0.0, 880.0), (1.0, 440.0)]
        result = render_click_track(sched, SR, click_s=0.05, click_amp=0.5)
        win_a  = result[: int(0.01 * SR)]
        win_b  = result[int(1.0 * SR): int(1.0 * SR) + int(0.01 * SR)]
        assert np.max(np.abs(win_a)) > 0.1
        assert np.max(np.abs(win_b)) > 0.1

    def test_output_covers_last_click_plus_click_s(self):
        click_s = 0.05
        last_t  = 2.0
        sched   = [(0.0, 440.0), (last_t, 440.0)]
        result  = render_click_track(sched, SR, click_s=click_s)
        min_samples = int((last_t + click_s) * SR)
        assert len(result) >= min_samples

    def test_samples_before_first_click_are_silent(self):
        # First click at 0.5s; samples before 0.4s should be zero.
        sched  = [(0.5, 440.0)]
        result = render_click_track(sched, SR, click_s=0.05, click_amp=0.5)
        pre    = result[: int(0.4 * SR)]
        assert np.max(np.abs(pre)) == pytest.approx(0.0)

    def test_amplitude_does_not_exceed_click_amp(self):
        sched  = click_schedule(60.0, 2, 4)
        amp    = 0.3
        result = render_click_track(sched, SR, click_amp=amp)
        # Allow a tiny floating-point margin.
        assert float(np.max(np.abs(result))) <= amp + 1e-5

    def test_empty_schedule_is_float32(self):
        result = render_click_track([], SR)
        assert result.dtype == np.float32


# ── Argument parsing — --click flag ──────────────────────────────────────────

class TestClickFlag:

    def test_default_click_is_false(self):
        assert _parse().click is False

    def test_click_flag_sets_true(self):
        assert _parse("--click").click is True

    def test_click_and_no_click_coexist(self):
        args = _parse("--click", "--no-click")
        assert args.click    is True
        assert args.no_click is True
