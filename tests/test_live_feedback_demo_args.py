"""Tests for scripts/live_feedback_demo.py argument parsing and output formatting.

sounddevice is mocked so no audio hardware is needed.  Only the pure-Python
logic (_parse_args and _format_event) is exercised here.

Test matrix
-----------
Argument parsing
    1.  Default namespace has expected values for all new and existing flags.
    2.  --adaptive-timing sets adaptive_timing=True.
    3.  --adaptive-window-shift overrides the default fraction.
    4.  --max-window-shift-beats overrides the default cap.
    5.  All three adaptive flags together are parsed correctly.
    6.  Pre-existing flags (--no-click, --level-check) still parse alongside new flags.
    7.  --adaptive-window-shift=0.0 is accepted (no shift, just scoring).
    8.  --adaptive-window-shift=1.0 is accepted (full shift to adjusted).

_format_event output
    9.  Fixed-grid event: no adaptive suffix in the output.
    10. Adaptive event with onset: suffix contains bpm and win fields.
    11. Adaptive event with positive window shift: win value is positive.
    12. Adaptive event with negative window shift: win value is negative.
    13. Adaptive event with zero window shift: win value is +0 ms.
    14. Adaptive miss event (no onset): suffix still present (tracker fields intact).
    15. Event message is included in the line when present.
    16. Event message absent → no trailing whitespace artefact.
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import pytest

# ── Import the demo module without triggering sounddevice hardware queries ────

# sounddevice is imported at module level in the demo; patch it before import.
_sd_mock = unittest.mock.MagicMock()
_sd_mock.query_devices.return_value = []

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with unittest.mock.patch.dict("sys.modules", {"sounddevice": _sd_mock}):
    # Fresh import with the mock in place
    import importlib
    import scripts.live_feedback_demo as _demo_mod
    importlib.reload(_demo_mod)  # ensure the patched module is live

_parse_args   = _demo_mod._parse_args
_format_event = _demo_mod._format_event


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(*argv: str) -> "argparse.Namespace":  # type: ignore[name-defined]
    """Call _parse_args with the given CLI tokens."""
    with unittest.mock.patch("sys.argv", ["live_feedback_demo.py", *argv]):
        return _parse_args()


def _make_eval(rms: float = 0.0, peak: float = 0.0) -> dict:
    return {
        "detected": False, "rms": rms, "peak": peak,
        "onset_found": False, "onset_sample": None, "onset_time_s": None,
    }


def _fixed_event(
    idx: int = 0,
    severity: str = "miss",
    timing_error_s: float | None = None,
    messages: list[str] | None = None,
) -> dict:
    return {
        "target_index":      idx,
        "evaluation":        _make_eval(),
        "severity":          severity,
        "timing_error_s":    timing_error_s,
        "messages":          messages or [],
        "expected_note":     None,
        "detected_note":     None,
        "pitch_error_cents": None,
        "confidence":        None,
    }


def _adaptive_event(
    idx: int = 0,
    severity: str = "good",
    timing_error_s: float = 0.010,
    current_bpm: float = 61.5,
    window_shift_s: float = -0.008,
    messages: list[str] | None = None,
) -> dict:
    ev = _fixed_event(idx, severity, timing_error_s, messages)
    ev["detected_note"] = "?"
    ev.update({
        "timing_grid":               "adaptive",
        "nominal_target_time_s":     2.0,
        "adjusted_target_time_s":    1.992,
        "window_center_time_s":      1.996,
        "window_shift_s":            window_shift_s,
        "tempo_ratio":               61.5 / 60.0,
        "current_bpm":               current_bpm,
        "tempo_tracker_confidence":  0.85,
    })
    return ev


# ── 1–8: Argument parsing ─────────────────────────────────────────────────────

def test_default_adaptive_timing_is_false():
    assert _parse().adaptive_timing is False


def test_default_adaptive_window_shift():
    assert _parse().adaptive_window_shift == pytest.approx(0.5)


def test_default_max_window_shift_beats():
    assert _parse().max_window_shift_beats == pytest.approx(0.30)


def test_default_device_is_none():
    assert _parse().device is None


def test_default_no_click_is_false():
    assert _parse().no_click is False


def test_adaptive_timing_flag_sets_true():
    assert _parse("--adaptive-timing").adaptive_timing is True


def test_adaptive_window_shift_override():
    args = _parse("--adaptive-timing", "--adaptive-window-shift", "0.25")
    assert args.adaptive_window_shift == pytest.approx(0.25)


def test_max_window_shift_beats_override():
    args = _parse("--adaptive-timing", "--max-window-shift-beats", "0.20")
    assert args.max_window_shift_beats == pytest.approx(0.20)


def test_all_adaptive_flags_together():
    args = _parse(
        "--adaptive-timing",
        "--adaptive-window-shift", "0.3",
        "--max-window-shift-beats", "0.20",
    )
    assert args.adaptive_timing is True
    assert args.adaptive_window_shift == pytest.approx(0.3)
    assert args.max_window_shift_beats == pytest.approx(0.20)


def test_existing_flags_still_parsed_alongside_adaptive():
    args = _parse("--no-click", "--adaptive-timing")
    assert args.no_click is True
    assert args.adaptive_timing is True


def test_level_check_and_adaptive_together():
    args = _parse("--level-check", "--adaptive-timing")
    assert args.level_check is True
    assert args.adaptive_timing is True


def test_adaptive_window_shift_zero_accepted():
    args = _parse("--adaptive-timing", "--adaptive-window-shift", "0.0")
    assert args.adaptive_window_shift == pytest.approx(0.0)


def test_adaptive_window_shift_one_accepted():
    args = _parse("--adaptive-timing", "--adaptive-window-shift", "1.0")
    assert args.adaptive_window_shift == pytest.approx(1.0)


# ── 9–16: _format_event output ────────────────────────────────────────────────

def test_fixed_event_no_adaptive_suffix():
    line = _format_event(_fixed_event(), t_now=3.0)
    assert "[bpm=" not in line
    assert "win=" not in line


def test_fixed_event_contains_beat_index():
    line = _format_event(_fixed_event(idx=2), t_now=3.0)
    assert "beat 2" in line


def test_fixed_event_contains_severity():
    line = _format_event(_fixed_event(severity="good"), t_now=3.0)
    assert "GOOD" in line


def test_fixed_event_timing_error_formatted():
    line = _format_event(_fixed_event(timing_error_s=0.025), t_now=3.0)
    assert "+25 ms" in line or "+25ms" in line.replace(" ", "")


def test_fixed_event_no_onset_shows_dashes():
    line = _format_event(_fixed_event(timing_error_s=None), t_now=3.0)
    assert "-- ms" in line


def test_adaptive_event_has_bpm_suffix():
    line = _format_event(_adaptive_event(current_bpm=61.5), t_now=3.0)
    assert "[bpm=61.5" in line


def test_adaptive_event_has_win_suffix():
    line = _format_event(_adaptive_event(window_shift_s=-0.008), t_now=3.0)
    assert "win=" in line


def test_adaptive_event_positive_shift_shows_plus():
    line = _format_event(_adaptive_event(window_shift_s=0.010), t_now=3.0)
    assert "win=+10ms" in line


def test_adaptive_event_negative_shift_shows_minus():
    line = _format_event(_adaptive_event(window_shift_s=-0.010), t_now=3.0)
    assert "win=-10ms" in line


def test_adaptive_event_zero_shift():
    line = _format_event(_adaptive_event(window_shift_s=0.0), t_now=3.0)
    assert "win=+0ms" in line


def test_adaptive_miss_event_has_suffix():
    ev = _adaptive_event(severity="miss", timing_error_s=None)
    ev["detected_note"] = None
    line = _format_event(ev, t_now=3.0)
    assert "[bpm=" in line


def test_message_included_in_line():
    ev = _fixed_event(messages=["Good timing"])
    line = _format_event(ev, t_now=3.0)
    assert "Good timing" in line


def test_no_message_no_trailing_spaces():
    ev = _fixed_event(messages=[])
    line = _format_event(ev, t_now=3.0)
    assert not line.endswith("  ")


def test_adaptive_suffix_comes_after_message():
    ev = _adaptive_event(messages=["Good timing"], window_shift_s=-0.005)
    line = _format_event(ev, t_now=3.0)
    msg_pos = line.index("Good timing")
    bpm_pos = line.index("[bpm=")
    assert msg_pos < bpm_pos
