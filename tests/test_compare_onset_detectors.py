"""Tests for compare_onset_detectors.py pure helpers.

Synthetic audio only — no audio hardware, no librosa required.

Test matrix
-----------
windowed_live_detect
    1.  All-silence buffer → all beats missed.
    2.  Clean attack exactly at beat → beat matched, error near 0 ms.
    3.  Clean attack at every beat → all beats matched.
    4.  Attack only at beat 0 → beat 0 matched, others missed.
    5.  Row count equals n_beats.
    6.  beat_s values match the expected grid times.
    7.  Attack 30 ms after beat → error_ms > 0 (late).
    8.  Attack 20 ms before beat (within pre-roll) → error_ms < 0 (early).

energy_derivative_detect
    9.  All-silence buffer → no onsets detected.
    10. Single clear attack → one onset within 50 ms of expected time.
    11. Two attacks separated by > min_gap → two onsets in chronological order.
    12. Amplitude below threshold → not detected.
    13. Two attacks within min_gap → at most one detected (gap enforced).
    14. Audio shorter than one frame → empty result.
    15. All onset times within [0, duration_s].

build_comparison_table
    16. Returns one row per beat (equals len(grid)).
    17. Detector columns present for every detector name.
    18. MISS row: <name>_ms is None, <name>_matched is False.
    19. Matched row: <name>_ms is float, <name>_matched is True.
    20. beat_idx and beat_s carried through correctly.

detector_summary
    21. All matched → beats_missed == 0, means are numeric.
    22. All missed → mean errors are None, beats_matched == 0.
    23. Partial match → counts correct.
    24. Mean signed error computed correctly (sign preserved).
    25. Mean absolute error computed correctly.
    26. Empty beat_rows → all zeros / None.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_onset_detectors import (
    windowed_live_detect,
    energy_derivative_detect,
    build_comparison_table,
    detector_summary,
)


# ── Shared constants ──────────────────────────────────────────────────────────

SR       = 16_000   # small sample rate keeps arrays fast
BPM      = 60.0
COUNT_IN = 1
N_BEATS  = 2
BEAT_S   = 60.0 / BPM          # 1.0 s

# Total audio: enough to cover all beats plus a tail.
TOTAL_S  = (COUNT_IN + N_BEATS + 0.5) * BEAT_S  # 3.5 s


# ── Synthetic audio helpers ───────────────────────────────────────────────────

def _silence() -> np.ndarray:
    return np.zeros(round(TOTAL_S * SR), dtype=np.float32)


def _beat_time(beat_idx: int) -> float:
    return COUNT_IN * BEAT_S + beat_idx * BEAT_S


def _add_attack(
    audio: np.ndarray,
    time_s: float,
    amplitude: float = 0.5,
    duration_s: float = 0.05,
) -> np.ndarray:
    """Return a copy of *audio* with a rectangular attack injected at *time_s*."""
    a     = audio.copy()
    start = round(time_s * SR)
    end   = min(len(a), start + round(duration_s * SR))
    a[start:end] = amplitude
    return a


# ── windowed_live_detect ──────────────────────────────────────────────────────

class TestWindowedLiveDetect:

    def test_silence_all_missed(self):
        rows = windowed_live_detect(_silence(), SR, BPM, COUNT_IN, N_BEATS)
        assert all(not r["matched"] for r in rows)

    def test_attack_at_beat_matched_near_zero(self):
        audio = _add_attack(_silence(), _beat_time(0))
        rows  = windowed_live_detect(audio, SR, BPM, COUNT_IN, N_BEATS)
        assert rows[0]["matched"] is True
        assert abs(rows[0]["error_ms"]) < 10.0  # within one hop

    def test_all_beats_attacked_all_matched(self):
        audio = _silence()
        for i in range(N_BEATS):
            audio = _add_attack(audio, _beat_time(i))
        rows = windowed_live_detect(audio, SR, BPM, COUNT_IN, N_BEATS)
        assert all(r["matched"] for r in rows)

    def test_only_first_beat_attacked(self):
        audio = _add_attack(_silence(), _beat_time(0))
        rows  = windowed_live_detect(audio, SR, BPM, COUNT_IN, N_BEATS)
        assert rows[0]["matched"] is True
        assert all(not r["matched"] for r in rows[1:])

    def test_row_count_equals_n_beats(self):
        rows = windowed_live_detect(_silence(), SR, BPM, COUNT_IN, N_BEATS)
        assert len(rows) == N_BEATS

    def test_beat_s_values_match_grid(self):
        rows = windowed_live_detect(_silence(), SR, BPM, COUNT_IN, N_BEATS)
        for i, row in enumerate(rows):
            assert row["beat_s"] == pytest.approx(COUNT_IN * BEAT_S + i * BEAT_S)

    def test_late_attack_positive_error(self):
        audio = _add_attack(_silence(), _beat_time(0) + 0.03)
        rows  = windowed_live_detect(audio, SR, BPM, COUNT_IN, N_BEATS)
        assert rows[0]["matched"] is True
        assert rows[0]["error_ms"] > 0

    def test_early_attack_within_preroll_negative_error(self):
        # 20 ms before beat, within the default 30 ms pre-roll.
        audio = _add_attack(_silence(), _beat_time(0) - 0.02)
        rows  = windowed_live_detect(audio, SR, BPM, COUNT_IN, N_BEATS)
        assert rows[0]["matched"] is True
        assert rows[0]["error_ms"] < 0

    def test_missed_beat_fields_are_none(self):
        rows = windowed_live_detect(_silence(), SR, BPM, COUNT_IN, N_BEATS)
        for row in rows:
            assert row["onset_s"]  is None
            assert row["error_ms"] is None


# ── energy_derivative_detect ──────────────────────────────────────────────────

class TestEnergyDerivativeDetect:

    # Detector params tuned for SR=16000 and amplitude=0.5.
    _THR = 0.05
    _HOP = 128

    def test_silence_no_onsets(self):
        onsets = energy_derivative_detect(_silence(), SR)
        assert len(onsets) == 0

    def test_single_attack_within_50ms(self):
        attack_t = 1.0
        audio    = _add_attack(_silence(), attack_t)
        onsets   = energy_derivative_detect(audio, SR, threshold=self._THR, hop_length=self._HOP)
        assert len(onsets) >= 1
        assert min(abs(t - attack_t) for t in onsets) < 0.05

    def test_two_attacks_two_onsets_in_order(self):
        audio = _add_attack(_silence(), 0.5)
        audio = _add_attack(audio,      2.0)
        onsets = energy_derivative_detect(
            audio, SR, threshold=self._THR, hop_length=self._HOP, min_gap_s=0.3
        )
        assert len(onsets) == 2
        assert onsets[0] < onsets[1]

    def test_below_threshold_not_detected(self):
        audio  = _add_attack(_silence(), 1.0, amplitude=0.001)
        onsets = energy_derivative_detect(audio, SR, threshold=0.5)
        assert len(onsets) == 0

    def test_min_gap_suppresses_duplicate(self):
        # Two attacks 10 ms apart; min_gap_s=0.10 → at most one detection.
        audio = _add_attack(_silence(), 1.00)
        audio = _add_attack(audio, 1.01, amplitude=0.4)
        onsets = energy_derivative_detect(
            audio, SR, threshold=self._THR, hop_length=self._HOP, min_gap_s=0.10
        )
        assert len(onsets) <= 1

    def test_audio_shorter_than_frame_returns_empty(self):
        audio  = np.zeros(100, dtype=np.float32)
        onsets = energy_derivative_detect(audio, SR, frame_length=512)
        assert len(onsets) == 0

    def test_onset_times_within_audio_duration(self):
        audio  = _add_attack(_silence(), 1.0)
        dur_s  = len(audio) / SR
        onsets = energy_derivative_detect(audio, SR, threshold=self._THR, hop_length=self._HOP)
        assert all(0.0 <= t <= dur_s for t in onsets)

    def test_onsets_are_sorted(self):
        audio = _add_attack(_silence(), 0.5)
        audio = _add_attack(audio,      2.0)
        onsets = energy_derivative_detect(
            audio, SR, threshold=self._THR, hop_length=self._HOP, min_gap_s=0.2
        )
        assert list(onsets) == sorted(onsets)


# ── build_comparison_table ────────────────────────────────────────────────────

def _make_beat_rows(errors: list[float | None]) -> list[dict]:
    return [
        {
            "beat_idx": i,
            "beat_s":   float(i + 1),
            "onset_s":  None if e is None else float(i + 1) + e / 1000.0,
            "error_ms": e,
            "matched":  e is not None,
        }
        for i, e in enumerate(errors)
    ]


class TestBuildComparisonTable:

    def test_row_count_equals_grid_length(self):
        grid  = np.array([1.0, 2.0, 3.0])
        a     = _make_beat_rows([10.0, None, -5.0])
        b     = _make_beat_rows([None, 20.0, -3.0])
        table = build_comparison_table(grid, [("A", a), ("B", b)])
        assert len(table) == 3

    def test_columns_present_for_each_detector(self):
        grid  = np.array([1.0])
        a     = _make_beat_rows([10.0])
        b     = _make_beat_rows([None])
        table = build_comparison_table(grid, [("det1", a), ("det2", b)])
        row   = table[0]
        for name in ("det1", "det2"):
            assert f"{name}_ms"      in row
            assert f"{name}_matched" in row

    def test_miss_row_none_and_false(self):
        grid  = np.array([1.0])
        a     = _make_beat_rows([None])
        table = build_comparison_table(grid, [("d", a)])
        assert table[0]["d_ms"]      is None
        assert table[0]["d_matched"] is False

    def test_matched_row_float_and_true(self):
        grid  = np.array([1.0])
        a     = _make_beat_rows([+25.0])
        table = build_comparison_table(grid, [("d", a)])
        assert table[0]["d_ms"]      == pytest.approx(25.0)
        assert table[0]["d_matched"] is True

    def test_beat_idx_and_beat_s_correct(self):
        grid  = np.array([1.0, 2.5])
        a     = _make_beat_rows([None, None])
        table = build_comparison_table(grid, [("d", a)])
        assert table[0]["beat_idx"] == 0
        assert table[0]["beat_s"]   == pytest.approx(1.0)
        assert table[1]["beat_idx"] == 1
        assert table[1]["beat_s"]   == pytest.approx(2.5)

    def test_multiple_detectors_independent_columns(self):
        """Detector A matched, detector B missed on same beat."""
        grid  = np.array([1.0])
        a     = _make_beat_rows([+10.0])
        b     = _make_beat_rows([None])
        table = build_comparison_table(grid, [("A", a), ("B", b)])
        row   = table[0]
        assert row["A_matched"] is True
        assert row["B_matched"] is False
        assert row["A_ms"] == pytest.approx(10.0)
        assert row["B_ms"] is None


# ── detector_summary ──────────────────────────────────────────────────────────

def _summary_rows(errors: list[float | None]) -> list[dict]:
    return [{"error_ms": e, "matched": e is not None} for e in errors]


class TestDetectorSummary:

    def test_all_matched(self):
        s = detector_summary(_summary_rows([10.0, -20.0]))
        assert s["beats_matched"] == 2
        assert s["beats_missed"]  == 0
        assert s["beats_total"]   == 2

    def test_all_missed_means_none(self):
        s = detector_summary(_summary_rows([None, None]))
        assert s["beats_matched"]        == 0
        assert s["beats_missed"]         == 2
        assert s["mean_signed_error_ms"] is None
        assert s["mean_abs_error_ms"]    is None

    def test_partial_match_counts(self):
        s = detector_summary(_summary_rows([10.0, None, -30.0]))
        assert s["beats_total"]   == 3
        assert s["beats_matched"] == 2
        assert s["beats_missed"]  == 1

    def test_mean_signed_error(self):
        # +20 and −40 → mean = −10
        s = detector_summary(_summary_rows([20.0, -40.0]))
        assert s["mean_signed_error_ms"] == pytest.approx(-10.0)

    def test_mean_absolute_error(self):
        # |+20| and |−40| → mean = 30
        s = detector_summary(_summary_rows([20.0, -40.0]))
        assert s["mean_abs_error_ms"] == pytest.approx(30.0)

    def test_empty_rows_all_zero_or_none(self):
        s = detector_summary([])
        assert s["beats_total"]          == 0
        assert s["beats_matched"]        == 0
        assert s["beats_missed"]         == 0
        assert s["mean_signed_error_ms"] is None
        assert s["mean_abs_error_ms"]    is None
