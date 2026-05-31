"""Tests for inspect-mode helpers and time-base properties in practice_replay_viewer.py.

All tests are pure — no audio hardware, no browser, no file I/O.

Test matrix
-----------
compute_inspect_view
    1.  Symmetric unclamped: center far from boundaries → start = center - half, end = center + half.
    2.  Clamped at 0: center near zero → start clamped to 0.0.
    3.  Clamped at total_duration: center near end → end clamped to total_duration.
    4.  ±10 ms window → half = 0.010 s.
    5.  Window larger than full duration → both ends clamped.
    6.  center_s == 0.0 → start = 0.0, end = half (or total_duration if smaller).
    7.  center_s == total_duration → end = total_duration, start = total_duration - half (or 0).

ruler_step_ms
    8.  window_ms == 10.0 → 10.0.
    9.  window_ms == 25.0 → 10.0 (boundary: ≤ 25 → fine scale).
    10. window_ms == 25.1 → 25.0 (just above boundary → coarse).
    11. window_ms == 50.0 → 25.0.
    12. window_ms == 100.0 → 25.0.
    13. window_ms == 1.0 → 10.0 (very small window).

ruler_ticks
    14. Center tick is always present and labeled "0".
    15. Positive ticks formatted "+X ms".
    16. Negative ticks formatted "X ms" (no plus).
    17. Only ticks within [view_start_s, view_end_s] are returned.
    18. Tick count matches expected count for ±50 ms / 25 ms step.
    19. Tick positions are spaced exactly step_ms apart.
    20. step_ms == 0 → empty list.
    21. Negative step_ms → empty list.
    22. Window wider than step: first tick ≥ view_start_s; last tick ≤ view_end_s.
    23. ±10 ms window / 10 ms step → exactly 3 ticks (-10, 0, +10).

time base (via extract_replay_data)
    24. target_times_s[0] == count_in_s  (first exercise beat = DOWNBEAT).
    25. target_times_s[i] == count_in_s + i * beat_s  for all i.
    26. First downbeat == count_in_beats × 60 / bpm.
    27. count-in clicks are at 0, beat_s, …, (count_in-1) × beat_s.
    28. Exercise clicks == target_times_s.
    29. corrected onset = target_s + err_s  (latency already removed).
    30. raw onset = corrected onset + latency_ms/1000.
    31. compute_inspect_view centered on target_times_s[0] with ±50 ms unclamped.
    32. compute_inspect_view centered on onset_s (corrected) with ±25 ms.

sample_index_to_s
    33. sample_index=0 → 0.0 s for any sample_rate.
    34. sample_index == sample_rate → 1.0 s exactly.
    35. sample_index=22050, sample_rate=44100 → 0.5 s.
    36. bpm=60, count_in=4 → target_times_s[0] == 4.0 s (count_in × beat_s = 4 × 1.0 s).
    37. play window (compute_inspect_view) centered on target: start < center < end.
    38. play window start/end values are clamped to [0, total_duration].
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT    = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from practice_replay_viewer import (
    compute_inspect_view,
    effective_view_duration,
    extract_replay_data,
    ruler_step_ms,
    ruler_ticks,
    sample_index_to_s,
)
from core.session_log import TARGET_HIT, TARGET_MISS, SessionEvent, SessionLog


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_log(metadata: dict | None = None, *events: SessionEvent) -> SessionLog:
    log = SessionLog(schema_version="1", started_at="2026-01-01T00:00:00+00:00")
    if metadata:
        log.metadata.update(metadata)
    log.events.extend(events)
    return log


def _hit(target_index: int, value_s: float, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=TARGET_HIT,
        target_index=target_index,
        value=value_s,
    )


def _miss(target_index: int, time_sec: float = 1.0) -> SessionEvent:
    return SessionEvent(
        time_sec=time_sec,
        event_type=TARGET_MISS,
        target_index=target_index,
        value=None,
    )


_META = {"bpm": "120", "beats": "4", "count_in": "2", "latency_ms": "0"}


# ── compute_inspect_view ──────────────────────────────────────────────────────

def test_inspect_view_symmetric_unclamped():
    start, end = compute_inspect_view(center_s=2.0, window_ms=100.0, total_duration=10.0)
    assert start == pytest.approx(1.9)
    assert end   == pytest.approx(2.1)


def test_inspect_view_clamped_at_zero():
    # center close to start; window would go negative
    start, end = compute_inspect_view(center_s=0.05, window_ms=100.0, total_duration=10.0)
    assert start == pytest.approx(0.0)
    assert end   == pytest.approx(0.15)


def test_inspect_view_clamped_at_total_duration():
    # center close to end; window would exceed total_duration
    start, end = compute_inspect_view(center_s=9.95, window_ms=100.0, total_duration=10.0)
    assert start == pytest.approx(9.85)
    assert end   == pytest.approx(10.0)


def test_inspect_view_10ms_window():
    start, end = compute_inspect_view(center_s=1.0, window_ms=10.0, total_duration=5.0)
    assert start == pytest.approx(0.990)
    assert end   == pytest.approx(1.010)


def test_inspect_view_window_larger_than_duration():
    # window huge → both ends clamped
    start, end = compute_inspect_view(center_s=0.5, window_ms=5000.0, total_duration=1.0)
    assert start == pytest.approx(0.0)
    assert end   == pytest.approx(1.0)


def test_inspect_view_center_at_zero():
    start, end = compute_inspect_view(center_s=0.0, window_ms=50.0, total_duration=5.0)
    assert start == pytest.approx(0.0)
    assert end   == pytest.approx(0.05)


def test_inspect_view_center_at_total_duration():
    start, end = compute_inspect_view(center_s=5.0, window_ms=50.0, total_duration=5.0)
    assert start == pytest.approx(4.95)
    assert end   == pytest.approx(5.0)


# ── ruler_step_ms ─────────────────────────────────────────────────────────────

def test_step_10ms_window():
    assert ruler_step_ms(10.0) == pytest.approx(10.0)


def test_step_25ms_boundary_fine():
    # exactly 25 ms → ≤ 25 → fine scale (10 ms)
    assert ruler_step_ms(25.0) == pytest.approx(10.0)


def test_step_above_25_coarse():
    assert ruler_step_ms(25.1) == pytest.approx(25.0)


def test_step_50ms():
    assert ruler_step_ms(50.0) == pytest.approx(25.0)


def test_step_100ms():
    assert ruler_step_ms(100.0) == pytest.approx(25.0)


def test_step_very_small_window():
    assert ruler_step_ms(1.0) == pytest.approx(10.0)


# ── ruler_ticks ───────────────────────────────────────────────────────────────

def test_ruler_center_tick_labeled_zero():
    # ±50 ms window, 25 ms step → ticks at -50, -25, 0, +25, +50
    ticks = ruler_ticks(
        view_start_s=0.95, view_end_s=1.05,
        center_s=1.0, step_ms=25.0,
    )
    times  = [t for t, _ in ticks]
    labels = [lbl for _, lbl in ticks]
    closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - 1.0))
    assert times[closest_idx] == pytest.approx(1.0)
    assert labels[closest_idx] == "0"


def test_ruler_positive_label_format():
    ticks = ruler_ticks(
        view_start_s=1.0, view_end_s=1.1,
        center_s=1.0, step_ms=25.0,
    )
    positive = [(t, lbl) for t, lbl in ticks if t > 1.0]
    assert len(positive) >= 1
    for _, lbl in positive:
        assert lbl.startswith("+")
        assert "ms" in lbl


def test_ruler_negative_label_format():
    ticks = ruler_ticks(
        view_start_s=0.9, view_end_s=1.0,
        center_s=1.0, step_ms=25.0,
    )
    negative = [(t, lbl) for t, lbl in ticks if t < 1.0]
    assert len(negative) >= 1
    for _, lbl in negative:
        assert lbl.startswith("-")
        assert "ms" in lbl


def test_ruler_all_ticks_within_view():
    start, end, center = 0.95, 1.05, 1.0
    ticks = ruler_ticks(start, end, center, step_ms=25.0)
    for t, _ in ticks:
        assert t >= start - 1e-9
        assert t <= end   + 1e-9


def test_ruler_count_50ms_window_25ms_step():
    # ±50 ms window → view = [0.95, 1.05], step = 25 ms
    # ticks at center-50ms, center-25ms, center, center+25ms, center+50ms → 5 ticks
    ticks = ruler_ticks(
        view_start_s=0.95, view_end_s=1.05,
        center_s=1.0, step_ms=25.0,
    )
    assert len(ticks) == 5


def test_ruler_10ms_window_10ms_step_three_ticks():
    # ±10 ms → view = [0.990, 1.010], step = 10 ms → -10, 0, +10 ms
    ticks = ruler_ticks(
        view_start_s=0.990, view_end_s=1.010,
        center_s=1.0, step_ms=10.0,
    )
    assert len(ticks) == 3
    labels = [lbl for _, lbl in ticks]
    assert labels == ["-10 ms", "0", "+10 ms"]


def test_ruler_tick_spacing():
    ticks = ruler_ticks(
        view_start_s=0.9, view_end_s=1.1,
        center_s=1.0, step_ms=25.0,
    )
    times = sorted(t for t, _ in ticks)
    for i in range(1, len(times)):
        assert times[i] - times[i - 1] == pytest.approx(0.025)


def test_ruler_step_zero_returns_empty():
    assert ruler_ticks(0.0, 1.0, 0.5, step_ms=0.0) == []


def test_ruler_negative_step_returns_empty():
    assert ruler_ticks(0.0, 1.0, 0.5, step_ms=-10.0) == []


def test_ruler_first_tick_gte_view_start():
    ticks = ruler_ticks(0.8, 1.2, center_s=1.0, step_ms=25.0)
    if ticks:
        assert ticks[0][0] >= 0.8 - 1e-9


def test_ruler_last_tick_lte_view_end():
    ticks = ruler_ticks(0.8, 1.2, center_s=1.0, step_ms=25.0)
    if ticks:
        assert ticks[-1][0] <= 1.2 + 1e-9


# ── Time base (via extract_replay_data) ───────────────────────────────────────

def test_first_target_equals_count_in_s():
    result = extract_replay_data(_make_log(_META))
    assert result["target_times_s"][0] == pytest.approx(result["count_in_s"])


def test_all_targets_spaced_beat_s_apart():
    result = extract_replay_data(_make_log(_META))
    beat_s  = result["beat_s"]
    ci_s    = result["count_in_s"]
    targets = result["target_times_s"]
    for i, t in enumerate(targets):
        assert t == pytest.approx(ci_s + i * beat_s)


def test_first_downbeat_equals_count_in_times_beat_s():
    # count_in_s = count_in * (60 / bpm)
    meta = {"bpm": "90", "beats": "4", "count_in": "3", "latency_ms": "0"}
    result   = extract_replay_data(_make_log(meta))
    expected = 3 * (60.0 / 90.0)
    assert result["target_times_s"][0] == pytest.approx(expected)
    assert result["count_in_s"]        == pytest.approx(expected)


def test_count_in_clicks_positions():
    # 120 BPM, 2 count-in → clicks at 0.0, 0.5
    result   = extract_replay_data(_make_log(_META))
    beat_s   = result["beat_s"]
    count_in = result["count_in"]
    all_clicks = result["click_times_s"]
    count_in_clicks = all_clicks[:count_in]
    expected = [i * beat_s for i in range(count_in)]
    assert count_in_clicks == pytest.approx(expected)


def test_exercise_clicks_equal_target_times():
    result        = extract_replay_data(_make_log(_META))
    count_in      = result["count_in"]
    exercise_clicks = result["click_times_s"][count_in:]
    targets         = result["target_times_s"]
    assert exercise_clicks == pytest.approx(targets)


def test_corrected_onset_equals_target_plus_err():
    err_s = 0.035
    log   = _make_log(_META, _hit(0, err_s))
    result = extract_replay_data(log)
    d      = result["onset_data"][0]
    assert d["onset_s"] == pytest.approx(d["target_s"] + err_s)


def test_raw_onset_equals_corrected_plus_latency():
    latency_ms = 80.0
    meta = {**_META, "latency_ms": str(latency_ms)}
    log  = _make_log(meta, _hit(0, 0.020))
    result = extract_replay_data(log)
    onset_s = result["onset_data"][0]["onset_s"]
    raw     = result["raw_onset_times_s"][0]
    assert raw == pytest.approx(onset_s + latency_ms / 1000.0)


def test_inspect_view_on_target_unclamped():
    # Target[0] for 120 BPM, count_in=2 is at 1.0 s; total_duration=3.5 s
    # ±50 ms → should not clamp
    result = extract_replay_data(_make_log(_META))
    center = result["target_times_s"][0]
    total  = result["total_duration"]
    start, end = compute_inspect_view(center, window_ms=50.0, total_duration=total)
    assert start == pytest.approx(center - 0.05)
    assert end   == pytest.approx(center + 0.05)


def test_inspect_view_on_corrected_onset():
    err_s = 0.030
    log   = _make_log(_META, _hit(1, err_s))
    result = extract_replay_data(log)
    onset_s = result["onset_data"][1]["onset_s"]
    total   = result["total_duration"]
    start, end = compute_inspect_view(onset_s, window_ms=25.0, total_duration=total)
    assert start == pytest.approx(onset_s - 0.025)
    assert end   == pytest.approx(onset_s + 0.025)


# ── sample_index_to_s ─────────────────────────────────────────────────────────

def test_sample_zero_is_time_zero():
    assert sample_index_to_s(0, 44100) == pytest.approx(0.0)


def test_sample_rate_samples_is_one_second():
    assert sample_index_to_s(44100, 44100) == pytest.approx(1.0)


def test_half_sample_rate_is_half_second():
    assert sample_index_to_s(22050, 44100) == pytest.approx(0.5)


def test_sample_index_at_48k():
    assert sample_index_to_s(48000, 48000) == pytest.approx(1.0)


def test_sample_index_arbitrary():
    # 1024 samples at 44100 Hz ≈ 0.023220 s
    assert sample_index_to_s(1024, 44100) == pytest.approx(1024 / 44100)


# ── Time base: bpm=60 / count_in=4 → target at 4.0 s ────────────────────────

def test_target_at_bpm60_count_in4_equals_4s():
    # 60 BPM → beat_s = 1.0 s; 4 count-in → count_in_s = 4.0 s
    meta = {"bpm": "60", "beats": "1", "count_in": "4", "latency_ms": "0"}
    result = extract_replay_data(_make_log(meta))
    assert result["target_times_s"][0] == pytest.approx(4.0)
    assert result["count_in_s"]        == pytest.approx(4.0)


# ── Play window (compute_inspect_view) properties ─────────────────────────────

def test_play_window_center_is_strictly_inside():
    center = 2.0
    start, end = compute_inspect_view(center, window_ms=100.0, total_duration=10.0)
    assert start < center < end


def test_play_window_clamped_start_is_zero():
    start, end = compute_inspect_view(center_s=0.03, window_ms=100.0, total_duration=5.0)
    assert start == pytest.approx(0.0)
    assert end   == pytest.approx(0.13)


def test_play_window_clamped_end_is_total_duration():
    total = 5.0
    start, end = compute_inspect_view(center_s=4.97, window_ms=100.0, total_duration=total)
    assert end   == pytest.approx(total)
    assert start == pytest.approx(4.87)


# ── effective_view_duration ───────────────────────────────────────────────────
# The planned duration comes from session metadata; the audio duration comes
# from the decoded WAV.  These differ when the session was stopped early.

def test_evd_uses_audio_when_shorter():
    # Session was stopped early — WAV shorter than planned
    assert effective_view_duration(6.0, 4.0) == pytest.approx(4.0)


def test_evd_uses_audio_when_longer():
    # WAV includes extra audio after planned end
    assert effective_view_duration(4.0, 6.0) == pytest.approx(6.0)


def test_evd_falls_back_when_no_audio():
    # No WAV embedded: audio_s is None
    assert effective_view_duration(6.0, None) == pytest.approx(6.0)


def test_evd_falls_back_when_zero_audio():
    # Degenerate WAV decoded to zero duration
    assert effective_view_duration(6.0, 0.0) == pytest.approx(6.0)


def test_evd_equal_durations():
    assert effective_view_duration(5.0, 5.0) == pytest.approx(5.0)


def test_evd_planned_duration_matches_extract_replay_data():
    # Verify that extract_replay_data total_duration is the planned value that
    # effective_view_duration receives as planned_s.
    # 120 BPM, 4 beats, 2 count-in → total = 1.0 + 4*0.5 + 0.5 = 3.5 s
    from core.session_log import SessionLog
    log = SessionLog(schema_version="1", started_at="2026-01-01T00:00:00+00:00")
    log.metadata.update({"bpm": "120", "beats": "4", "count_in": "2", "latency_ms": "0"})
    result = extract_replay_data(log)
    planned = result["total_duration"]
    assert planned == pytest.approx(3.5)
    # With a 3-second WAV, effective duration should be 3.0 s
    assert effective_view_duration(planned, 3.0) == pytest.approx(3.0)
    # With no audio, falls back to planned
    assert effective_view_duration(planned, None) == pytest.approx(planned)
