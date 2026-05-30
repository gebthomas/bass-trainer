"""Tests for waveform-plot helpers in scripts/practice_session_demo.py.

All tests are pure — no audio hardware, no matplotlib display.

Test matrix
-----------
_onset_plot_data
    1.  Empty timing_errors → three empty lists.
    2.  Single hit, latency=0 → onset in compensated; raw is empty.
    3.  Single hit, latency=100ms → onset in compensated; raw = onset + 0.1s.
    4.  Single miss → expected time appears in miss_target_times_s.
    5.  Mixed hits and misses produce correct per-list membership.
    6.  idx >= len(target_times_s) is silently skipped.
    7.  Exact-timing hit (err_ms=0) → onset equals expected time.
    8.  Negative timing error (early onset) → onset < expected time.
    9.  Zero latency never populates raw_onsets list.
    10. Non-zero latency: raw = compensated + offset for every hit.

_split_click_times
    11. Empty click list → both groups empty.
    12. All times before count_in_s → all in count-in group, exercise empty.
    13. All times at or after count_in_s → count-in empty, all in exercise.
    14. Times exactly equal to count_in_s go to exercise group (t >= count_in_s).
    15. count_in_s == 0.0 → all times go to exercise (no count-in window).
    16. Mixed times split correctly at boundary.
    17. Round-trip: click_beat_times_s output splits into expected counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT    = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from practice_session_demo import _onset_plot_data, _split_click_times, click_beat_times_s


# Shorthand for readable test data
# timing_errors element: (target_index, timing_error_ms | None)

def _run(
    errors: list[tuple[int, float | None]],
    targets: list[float],
    latency_ms: float = 0.0,
):
    return _onset_plot_data(errors, targets, latency_ms)


# ── _onset_plot_data ──────────────────────────────────────────────────────────

def test_empty_errors_returns_three_empty_lists():
    compensated, raw, misses = _run([], [1.0, 2.0])
    assert compensated == []
    assert raw == []
    assert misses == []


def test_single_hit_zero_latency_raw_is_empty():
    compensated, raw, misses = _run([(0, 50.0)], [1.0], latency_ms=0.0)
    assert len(compensated) == 1
    assert raw == []
    assert misses == []


def test_single_hit_compensated_onset_position():
    # target at 1.0s, 50ms late → onset at 1.050s
    compensated, _, _ = _run([(0, 50.0)], [1.0], latency_ms=0.0)
    assert compensated == pytest.approx([1.050])


def test_single_hit_with_latency_raw_populated():
    # target at 1.0s, 50ms late, latency=100ms
    # compensated onset = 1.050s; raw = 1.050 + 0.100 = 1.150s
    compensated, raw, _ = _run([(0, 50.0)], [1.0], latency_ms=100.0)
    assert compensated == pytest.approx([1.050])
    assert raw == pytest.approx([1.150])


def test_single_miss_appears_in_miss_list():
    _, _, misses = _run([(0, None)], [2.5])
    assert misses == pytest.approx([2.5])


def test_single_miss_absent_from_hit_lists():
    compensated, raw, _ = _run([(0, None)], [2.5], latency_ms=100.0)
    assert compensated == []
    assert raw == []


def test_mixed_hit_and_miss():
    targets = [1.0, 2.0, 3.0]
    errors  = [(0, 20.0), (1, None), (2, -30.0)]
    compensated, raw, misses = _run(errors, targets, latency_ms=0.0)
    assert compensated == pytest.approx([1.020, 2.970])
    assert misses == pytest.approx([2.0])
    assert raw == []


def test_out_of_range_idx_skipped():
    # idx=5 but only 3 targets → should be silently dropped
    errors = [(5, 10.0), (0, 20.0)]
    targets = [0.5, 1.0, 1.5]
    compensated, _, _ = _run(errors, targets)
    assert len(compensated) == 1
    assert compensated == pytest.approx([0.520])


def test_exact_timing_hit_onset_equals_expected():
    targets = [1.0]
    compensated, _, _ = _run([(0, 0.0)], targets)
    assert compensated == pytest.approx([1.0])


def test_negative_timing_error_onset_before_expected():
    targets = [2.0]
    compensated, _, _ = _run([(0, -40.0)], targets)
    assert compensated == pytest.approx([1.960])


def test_zero_latency_never_populates_raw():
    errors  = [(0, 10.0), (1, 20.0), (2, -5.0)]
    targets = [0.5, 1.0, 1.5]
    _, raw, _ = _run(errors, targets, latency_ms=0.0)
    assert raw == []


def test_nonzero_latency_raw_equals_compensated_plus_offset():
    latency_ms = 150.0
    targets = [1.0, 2.0]
    errors  = [(0, 30.0), (1, -20.0)]
    compensated, raw, _ = _run(errors, targets, latency_ms=latency_ms)
    offset_s = latency_ms / 1000.0
    assert raw == pytest.approx([c + offset_s for c in compensated])


# ── _split_click_times ────────────────────────────────────────────────────────

def test_split_empty_list():
    count_in, exercise = _split_click_times([], count_in_s=1.0)
    assert count_in == []
    assert exercise == []


def test_split_all_before_count_in():
    # Three count-in clicks at 120 BPM, beat_s=0.5s, count_in_s=1.5s
    times = [0.0, 0.5, 1.0]
    count_in, exercise = _split_click_times(times, count_in_s=1.5)
    assert count_in == pytest.approx([0.0, 0.5, 1.0])
    assert exercise == []


def test_split_all_at_or_after_count_in():
    times = [1.0, 1.5, 2.0]
    count_in, exercise = _split_click_times(times, count_in_s=1.0)
    assert count_in == []
    assert exercise == pytest.approx([1.0, 1.5, 2.0])


def test_split_exactly_at_boundary_goes_to_exercise():
    # A time exactly equal to count_in_s must land in exercise (t >= count_in_s)
    times = [0.9, 1.0, 1.1]
    count_in, exercise = _split_click_times(times, count_in_s=1.0)
    assert count_in == pytest.approx([0.9])
    assert exercise == pytest.approx([1.0, 1.1])


def test_split_zero_count_in_all_go_to_exercise():
    # count_in_s == 0 → no count-in window; every click is an exercise click
    times = [0.0, 0.5, 1.0]
    count_in, exercise = _split_click_times(times, count_in_s=0.0)
    assert count_in == []
    assert exercise == pytest.approx([0.0, 0.5, 1.0])


def test_split_mixed_times():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    count_in, exercise = _split_click_times(times, count_in_s=1.0)
    assert count_in  == pytest.approx([0.0, 0.5])
    assert exercise  == pytest.approx([1.0, 1.5, 2.0])


def test_split_roundtrip_with_click_beat_times_s():
    # click_beat_times_s(120 BPM, 2 count-in, [0,1,2,3]) →
    #   beat_s=0.5, count_in_s=1.0
    #   count-in: [0.0, 0.5]
    #   exercise: [1.0, 1.5, 2.0, 2.5]
    bpm, n_count_in = 120.0, 2
    target_positions = [0.0, 1.0, 2.0, 3.0]
    beat_s    = 60.0 / bpm
    count_in_s = n_count_in * beat_s

    times = click_beat_times_s(bpm, n_count_in, target_positions)
    count_in, exercise = _split_click_times(times, count_in_s)

    assert len(count_in) == n_count_in
    assert len(exercise)  == len(target_positions)
    assert count_in  == pytest.approx([i * beat_s for i in range(n_count_in)])
    assert exercise  == pytest.approx([count_in_s + p * beat_s for p in target_positions])
