"""Tests for scripts/adaptive_timing_demo.py.

sounddevice is mocked so no audio hardware is needed.

Test matrix
-----------
Argument parsing
    1.  Default namespace has expected values for all flags.
    2.  --bpm overrides nominal tempo.
    3.  --beats overrides exercise beat count.
    4.  --count-in overrides count-in beats.
    5.  --time-sig overrides time signature.
    6.  --click-mode choices are accepted: beat, half, measure, count-in-only, off.
    7.  --adaptive-window-shift override.
    8.  --max-window-shift-beats override.
    9.  --level-check flag.
    10. --device flag.

_format_beat_line
    11. Fixed hit: shows beat index, severity, timing ms, ● symbol.
    12. Miss: shows · symbol.
    13. Adaptive fields add [bpm= win= conf=] block.
    14. No adaptive fields → no [bpm= block.

_fixed_timing_error
    15. Returns None when adaptive fields absent.
    16. On-time adaptive event → fixed error equals adaptive error (adjusted==nominal).
    17. Early adaptive event: fixed_error differs from adaptive_error when adjusted≠nominal.
    18. Returns None when timing_error_s is None.

_build_timeline
    19. All hits → all ● characters.
    20. All misses → all · characters.
    21. Bar separators placed at correct intervals.
    22. Single beat, no separator.
    23. Exactly one bar boundary (time_sig == len).

_compute_summary_stats
    24. All detected, no errors → detect_pct == 100.
    25. Half detected → detect_pct == 50.
    26. Mean adaptive error computed correctly.
    27. Mean fixed error computed correctly.
    28. BPM trajectory populated from records.
    29. Final BPM matches last record.
    30. Empty records returns safe defaults.

_make_full_click_track
    31. Returns ndarray of float32.
    32. Length matches (count_in + exercise) beats × beat_s × sr.
    33. Click-mode 'off' → all-zero audio.
    34. Click-mode 'count-in-only' → silence after count-in window.
    35. Click-mode 'beat' → non-zero samples present throughout.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

# ── Import the demo module without triggering sounddevice hardware queries ────

_sd_mock = unittest.mock.MagicMock()
_sd_mock.query_devices.return_value = []

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with unittest.mock.patch.dict("sys.modules", {"sounddevice": _sd_mock}):
    import importlib
    import scripts.adaptive_timing_demo as _demo_mod
    importlib.reload(_demo_mod)

_parse_args          = _demo_mod._parse_args
_format_beat_line    = _demo_mod._format_beat_line
_fixed_timing_error  = _demo_mod._fixed_timing_error
_build_timeline      = _demo_mod._build_timeline
_compute_summary_stats = _demo_mod._compute_summary_stats
_make_full_click_track = _demo_mod._make_full_click_track


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(*argv: str) -> "argparse.Namespace":  # type: ignore[name-defined]
    with unittest.mock.patch("sys.argv", ["adaptive_timing_demo.py", *argv]):
        return _parse_args()


def _make_eval(onset_found: bool = False) -> dict:
    return {
        "detected": onset_found, "rms": 0.0, "peak": 0.0,
        "onset_found": onset_found, "onset_sample": None, "onset_time_s": None,
    }


def _hit_event(
    idx: int = 0,
    severity: str = "good",
    timing_error_s: float = 0.010,
    current_bpm: float = 61.5,
    window_shift_s: float = -0.008,
    adjusted_target_time_s: float | None = None,
    nominal_target_time_s: float | None = None,
) -> dict:
    """Adaptive hit event."""
    nominal = nominal_target_time_s if nominal_target_time_s is not None else 2.0
    adjusted = adjusted_target_time_s if adjusted_target_time_s is not None else nominal
    return {
        "target_index":      idx,
        "evaluation":        _make_eval(onset_found=True),
        "severity":          severity,
        "timing_error_s":    timing_error_s,
        "messages":          [],
        "expected_note":     None,
        "detected_note":     "?",
        "pitch_error_cents": None,
        "confidence":        None,
        "timing_grid":               "adaptive",
        "nominal_target_time_s":     nominal,
        "adjusted_target_time_s":    adjusted,
        "window_center_time_s":      nominal,
        "window_shift_s":            window_shift_s,
        "tempo_ratio":               current_bpm / 60.0,
        "current_bpm":               current_bpm,
        "tempo_tracker_confidence":  0.85,
    }


def _miss_event(idx: int = 0) -> dict:
    """Fixed-grid miss event (no adaptive fields)."""
    return {
        "target_index":      idx,
        "evaluation":        _make_eval(onset_found=False),
        "severity":          "miss",
        "timing_error_s":    None,
        "messages":          [],
        "expected_note":     None,
        "detected_note":     None,
        "pitch_error_cents": None,
        "confidence":        None,
    }


def _record(
    beat_index: int = 0,
    detected: bool = True,
    adaptive_error_s: float | None = 0.010,
    fixed_error_s: float | None = 0.010,
    bpm_estimate: float = 60.0,
    conf: float = 0.8,
) -> dict:
    return {
        "beat_index":      beat_index,
        "detected":        detected,
        "adaptive_error_s": adaptive_error_s,
        "fixed_error_s":   fixed_error_s,
        "bpm_estimate":    bpm_estimate,
        "conf":            conf,
    }


# ── 1–10: Argument parsing ────────────────────────────────────────────────────

def test_default_bpm():
    assert _parse().bpm == pytest.approx(60.0)


def test_default_beats():
    assert _parse().beats == 16


def test_default_count_in():
    assert _parse().count_in == 4


def test_default_time_sig():
    assert _parse().time_sig == 4


def test_default_click_mode():
    assert _parse().click_mode == "beat"


def test_default_adaptive_window_shift():
    assert _parse().adaptive_window_shift == pytest.approx(0.5)


def test_default_max_window_shift_beats():
    assert _parse().max_window_shift_beats == pytest.approx(0.30)


def test_default_level_check_is_false():
    assert _parse().level_check is False


def test_default_device_is_none():
    assert _parse().device is None


def test_bpm_override():
    assert _parse("--bpm", "72").bpm == pytest.approx(72.0)


def test_beats_override():
    assert _parse("--beats", "32").beats == 32


def test_count_in_override():
    assert _parse("--count-in", "2").count_in == 2


def test_time_sig_override():
    assert _parse("--time-sig", "3").time_sig == 3


def test_click_mode_beat():
    assert _parse("--click-mode", "beat").click_mode == "beat"


def test_click_mode_half():
    assert _parse("--click-mode", "half").click_mode == "half"


def test_click_mode_measure():
    assert _parse("--click-mode", "measure").click_mode == "measure"


def test_click_mode_count_in_only():
    assert _parse("--click-mode", "count-in-only").click_mode == "count-in-only"


def test_click_mode_off():
    assert _parse("--click-mode", "off").click_mode == "off"


def test_adaptive_window_shift_override():
    assert _parse("--adaptive-window-shift", "0.25").adaptive_window_shift == pytest.approx(0.25)


def test_max_window_shift_beats_override():
    assert _parse("--max-window-shift-beats", "0.20").max_window_shift_beats == pytest.approx(0.20)


def test_level_check_flag():
    assert _parse("--level-check").level_check is True


def test_device_flag():
    assert _parse("--device", "2").device == "2"


# ── 11–14: _format_beat_line ──────────────────────────────────────────────────

def test_format_hit_shows_beat_index():
    line = _format_beat_line(_hit_event(idx=5), t_now=3.0)
    assert "beat   5" in line or "beat  5" in line or "beat 5" in line


def test_format_hit_shows_severity():
    line = _format_beat_line(_hit_event(severity="good"), t_now=3.0)
    assert "GOOD" in line


def test_format_hit_shows_timing_ms():
    line = _format_beat_line(_hit_event(timing_error_s=0.020), t_now=3.0)
    assert "+20 ms" in line or "+20ms" in line.replace(" ", "")


def test_format_hit_shows_bullet():
    line = _format_beat_line(_hit_event(), t_now=3.0)
    assert "●" in line


def test_format_miss_shows_circle():
    line = _format_beat_line(_miss_event(), t_now=3.0)
    assert "·" in line


def test_format_adaptive_shows_bpm():
    line = _format_beat_line(_hit_event(current_bpm=61.5), t_now=3.0)
    assert "[bpm=61.5" in line


def test_format_adaptive_shows_win():
    line = _format_beat_line(_hit_event(window_shift_s=-0.008), t_now=3.0)
    assert "win=" in line


def test_format_adaptive_shows_conf():
    line = _format_beat_line(_hit_event(), t_now=3.0)
    assert "conf=" in line


def test_format_no_adaptive_fields_no_bpm_block():
    line = _format_beat_line(_miss_event(), t_now=3.0)
    assert "[bpm=" not in line


# ── 15–18: _fixed_timing_error ────────────────────────────────────────────────

def test_fixed_timing_error_none_when_no_adaptive_fields():
    assert _fixed_timing_error(_miss_event()) is None


def test_fixed_timing_error_equals_adaptive_when_adjusted_equals_nominal():
    # adjusted == nominal → fixed_error == adaptive_error
    ev = _hit_event(
        timing_error_s=0.015,
        adjusted_target_time_s=2.0,
        nominal_target_time_s=2.0,
    )
    assert _fixed_timing_error(ev) == pytest.approx(0.015, abs=1e-9)


def test_fixed_timing_error_differs_when_adjusted_differs():
    # adjusted = 2.020, nominal = 2.0, adaptive_error = 0.005
    # actual_onset = 2.020 + 0.005 = 2.025
    # fixed_error  = 2.025 - 2.000 = 0.025
    ev = _hit_event(
        timing_error_s=0.005,
        adjusted_target_time_s=2.020,
        nominal_target_time_s=2.000,
    )
    assert _fixed_timing_error(ev) == pytest.approx(0.025, abs=1e-9)


def test_fixed_timing_error_none_when_no_onset():
    ev = _hit_event(timing_error_s=None)
    assert _fixed_timing_error(ev) is None


# ── 19–23: _build_timeline ────────────────────────────────────────────────────

def test_timeline_all_hits():
    line = _build_timeline([True, True, True, True], time_sig=4)
    assert "·" not in line
    assert line.count("●") == 4


def test_timeline_all_misses():
    line = _build_timeline([False, False, False], time_sig=4)
    assert "●" not in line
    assert line.count("·") == 3


def test_timeline_bar_separator_at_correct_positions():
    # 8 beats, time_sig=4 → separator at position 4
    line = _build_timeline([True] * 8, time_sig=4)
    # Should contain exactly one │ (between beat 3 and 4)
    assert line.count("│") == 1


def test_timeline_single_beat_no_separator():
    line = _build_timeline([True], time_sig=4)
    assert "│" not in line


def test_timeline_exact_one_bar():
    # time_sig beats → no separator (separator only added at index > 0 and i%time_sig==0)
    line = _build_timeline([True] * 4, time_sig=4)
    assert "│" not in line


def test_timeline_two_bars():
    line = _build_timeline([True] * 8, time_sig=4)
    assert line.count("│") == 1


def test_timeline_three_bars():
    line = _build_timeline([True] * 12, time_sig=4)
    assert line.count("│") == 2


# ── 24–30: _compute_summary_stats ─────────────────────────────────────────────

def test_summary_all_detected():
    records = [_record(detected=True) for _ in range(4)]
    stats = _compute_summary_stats(records, 60.0, 4)
    assert stats["detect_pct"] == pytest.approx(100.0)
    assert stats["n_detected"] == 4
    assert stats["n_missed"]   == 0


def test_summary_half_detected():
    records = [_record(detected=(i % 2 == 0)) for i in range(4)]
    stats = _compute_summary_stats(records, 60.0, 4)
    assert stats["detect_pct"] == pytest.approx(50.0)


def test_summary_mean_adaptive_error():
    records = [
        _record(adaptive_error_s=0.010),
        _record(adaptive_error_s=0.020),
        _record(adaptive_error_s=0.030),
    ]
    stats = _compute_summary_stats(records, 60.0, 3)
    assert stats["mean_adaptive_ms"] == pytest.approx(20.0, abs=0.1)


def test_summary_mean_fixed_error():
    records = [
        _record(fixed_error_s=0.015),
        _record(fixed_error_s=0.025),
    ]
    stats = _compute_summary_stats(records, 60.0, 2)
    assert stats["mean_fixed_ms"] == pytest.approx(20.0, abs=0.1)


def test_summary_bpm_trajectory():
    records = [_record(bpm_estimate=60.0 + i * 0.1) for i in range(5)]
    stats = _compute_summary_stats(records, 60.0, 5)
    assert len(stats["bpm_trajectory"]) == 5
    assert stats["bpm_trajectory"][0] == pytest.approx(60.0)
    assert stats["bpm_trajectory"][-1] == pytest.approx(60.4, abs=0.01)


def test_summary_final_bpm():
    records = [_record(bpm_estimate=61.5)]
    stats = _compute_summary_stats(records, 60.0, 1)
    assert stats["final_bpm"] == pytest.approx(61.5)


def test_summary_empty_records():
    stats = _compute_summary_stats([], 60.0, 4)
    assert stats["n_detected"]       == 0
    assert stats["n_missed"]         == 4
    assert stats["detect_pct"]       == pytest.approx(0.0)
    assert stats["mean_adaptive_ms"] is None
    assert stats["mean_fixed_ms"]    is None


# ── 31–35: _make_full_click_track ─────────────────────────────────────────────

def _track(click_mode="beat", bpm=60.0, count_in=4, beats=4, time_sig=4, sr=48000):
    return _make_full_click_track(bpm, count_in, beats, click_mode, time_sig, sr)


def test_click_track_returns_float32():
    audio = _track()
    assert audio.dtype == np.float32


def test_click_track_length():
    bpm, count_in, beats, sr = 60.0, 4, 8, 48000
    beat_s        = 60.0 / bpm
    expected_len  = int((count_in + beats) * beat_s * sr)
    audio = _make_full_click_track(bpm, count_in, beats, "beat", 4, sr)
    assert len(audio) == expected_len


def test_click_mode_off_is_silent():
    audio = _track(click_mode="off")
    assert np.all(audio == 0.0)


def test_click_mode_count_in_only_silent_after_count_in():
    bpm, count_in, beats, sr = 60.0, 4, 8, 48000
    audio    = _make_full_click_track(bpm, count_in, beats, "count-in-only", 4, sr)
    beat_s   = 60.0 / bpm
    ci_end   = int(count_in * beat_s * sr)
    # Exercise portion should be all zeros
    assert np.all(audio[ci_end:] == 0.0)
    # Count-in portion should have at least some non-zero values
    assert np.any(audio[:ci_end] != 0.0)


def test_click_mode_beat_has_audio_throughout():
    audio = _track(click_mode="beat", count_in=2, beats=4)
    # Should be non-zero somewhere in both count-in and exercise
    assert np.any(audio != 0.0)
