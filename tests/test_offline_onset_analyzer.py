"""Tests for match_beats_to_onsets and beat_grid in offline_onset_analyzer.py.

Test matrix
-----------
Basic matching
    1.  Single beat, single onset within window → matched.
    2.  Single beat, no onset within window → missed.
    3.  Multiple beats, one-to-one matching, all matched.
    4.  Empty grid → no beat rows, summary zeros.
    5.  Empty onsets → all beats missed.
    6.  Both empty → no beat rows, zero counts.

Greedy one-to-one (key requirement)
    7.  Two onsets near same beat → only closer one matched, farther is unmatched.
    8.  Two beats near same onset → only closer one matched, other beat missed.
    9.  Closest pair is assigned first across competing beats/onsets.

Missed-beat / unmatched-onset reporting
    10. Missed beats appear in beat_rows with matched=False, onset_s=None, error_ms=None.
    11. Unmatched onsets appear in unmatched_onset_times.
    12. Onset just outside window → not matched.
    13. Onset just inside window → matched.

Summary statistics
    14. beats_total, beats_matched, beats_missed counts are correct.
    15. unmatched_onsets count is correct.
    16. mean_signed_error_ms and mean_abs_error_ms correct for matched rows.
    17. Both mean values are None when no beats are matched.

beat_grid helper
    18. Returns correct number of elements.
    19. First element equals count_in_s.
    20. Spacing between consecutive elements equals beat_s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.offline_onset_analyzer import beat_grid, match_beats_to_onsets


# ── helpers ───────────────────────────────────────────────────────────────────

def _grid(*times: float) -> np.ndarray:
    return np.array(times, dtype=np.float64)


def _onsets(*times: float) -> np.ndarray:
    return np.array(times, dtype=np.float64)


WIN = 0.15  # default window for most tests


# ── Basic matching ─────────────────────────────────────────────────────────────

class TestBasicMatching:

    def test_single_beat_single_onset_within_window(self):
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.05), WIN)
        row = result["beat_rows"][0]
        assert row["matched"] is True
        assert row["onset_s"] == pytest.approx(1.05)
        assert row["error_ms"] == pytest.approx(50.0)

    def test_single_beat_no_onset_within_window(self):
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.5), WIN)
        row = result["beat_rows"][0]
        assert row["matched"] is False
        assert row["onset_s"] is None
        assert row["error_ms"] is None

    def test_multiple_beats_all_matched(self):
        grid   = _grid(1.0, 2.0, 3.0)
        onsets = _onsets(1.02, 1.98, 3.01)
        result = match_beats_to_onsets(grid, onsets, WIN)
        rows = result["beat_rows"]
        assert all(r["matched"] for r in rows)
        assert rows[0]["error_ms"] == pytest.approx(20.0)
        assert rows[1]["error_ms"] == pytest.approx(-20.0)
        assert rows[2]["error_ms"] == pytest.approx(10.0)

    def test_empty_grid(self):
        result = match_beats_to_onsets(_grid(), _onsets(1.0, 2.0), WIN)
        assert result["beat_rows"] == []
        s = result["summary"]
        assert s["beats_total"]      == 0
        assert s["beats_matched"]    == 0
        assert s["beats_missed"]     == 0
        assert s["unmatched_onsets"] == 2

    def test_empty_onsets(self):
        result = match_beats_to_onsets(_grid(1.0, 2.0), _onsets(), WIN)
        rows = result["beat_rows"]
        assert all(not r["matched"] for r in rows)
        s = result["summary"]
        assert s["beats_missed"] == 2

    def test_both_empty(self):
        result = match_beats_to_onsets(_grid(), _onsets(), WIN)
        assert result["beat_rows"]             == []
        assert result["unmatched_onset_times"] == []
        s = result["summary"]
        assert s["beats_total"]   == 0
        assert s["beats_matched"] == 0


# ── Greedy one-to-one ─────────────────────────────────────────────────────────

class TestGreedyOneToOne:

    def test_two_onsets_near_same_beat_closer_wins(self):
        """Two onsets (0.02s and 0.08s from beat) — only 0.02s onset is matched."""
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.02, 1.08), WIN)
        row = result["beat_rows"][0]
        assert row["matched"] is True
        assert row["error_ms"] == pytest.approx(20.0)  # closer onset matched
        unmatched = result["unmatched_onset_times"]
        assert len(unmatched) == 1
        assert unmatched[0] == pytest.approx(1.08)

    def test_two_onsets_near_same_beat_both_in_window(self):
        """Both onsets within window; farther one ends up unmatched."""
        result = match_beats_to_onsets(_grid(2.0), _onsets(1.95, 2.10), WIN)
        row = result["beat_rows"][0]
        assert row["matched"] is True
        # 1.95 → |err|=0.05; 2.10 → |err|=0.10 — 1.95 wins
        assert row["error_ms"] == pytest.approx(-50.0)
        assert len(result["unmatched_onset_times"]) == 1
        assert result["unmatched_onset_times"][0] == pytest.approx(2.10)

    def test_two_beats_near_same_onset_closer_wins(self):
        """Onset at 1.05: beat at 1.0 (|err|=0.05) and beat at 1.12 (|err|=0.07)."""
        result = match_beats_to_onsets(_grid(1.0, 1.12), _onsets(1.05), WIN)
        rows = result["beat_rows"]
        # Beat 0 wins (0.05 < 0.07)
        assert rows[0]["matched"] is True
        assert rows[0]["error_ms"] == pytest.approx(50.0)
        # Beat 1 is missed
        assert rows[1]["matched"] is False
        assert result["unmatched_onset_times"] == []

    def test_greedy_cross_assignment(self):
        """Three beats, three onsets; greedy picks globally closest first.

        grid   = [1.0, 2.0, 3.0]
        onsets = [1.01, 2.05, 3.10]
        All within window; each should match its natural partner.
        """
        result = match_beats_to_onsets(_grid(1.0, 2.0, 3.0),
                                        _onsets(1.01, 2.05, 3.10), WIN)
        rows = result["beat_rows"]
        assert all(r["matched"] for r in rows)
        assert rows[0]["error_ms"] == pytest.approx(10.0)
        assert rows[1]["error_ms"] == pytest.approx(50.0)
        assert rows[2]["error_ms"] == pytest.approx(100.0)


# ── Missed-beat / unmatched-onset reporting ───────────────────────────────────

class TestMissedAndUnmatched:

    def test_missed_beat_row_fields(self):
        result = match_beats_to_onsets(_grid(1.0), _onsets(2.0), WIN)
        row = result["beat_rows"][0]
        assert row["matched"]   is False
        assert row["onset_s"]   is None
        assert row["error_ms"]  is None
        assert row["beat_idx"]  == 0
        assert row["beat_s"]    == pytest.approx(1.0)

    def test_unmatched_onset_in_list(self):
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.02, 5.0), WIN)
        assert result["beat_rows"][0]["matched"] is True
        unmatched = result["unmatched_onset_times"]
        assert len(unmatched) == 1
        assert unmatched[0] == pytest.approx(5.0)

    def test_onset_just_outside_window_not_matched(self):
        win = 0.10
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.101), win)
        assert result["beat_rows"][0]["matched"] is False

    def test_onset_inside_window_matched(self):
        # Use 99ms error with a 100ms window — unambiguously inside.
        win = 0.10
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.099), win)
        assert result["beat_rows"][0]["matched"] is True
        assert result["beat_rows"][0]["error_ms"] == pytest.approx(99.0)

    def test_mixed_matched_and_missed(self):
        grid   = _grid(1.0, 2.0, 3.0)
        onsets = _onsets(1.05, 5.0)   # beat 2 missed; onset at 5.0 unmatched
        result = match_beats_to_onsets(grid, onsets, WIN)
        rows = result["beat_rows"]
        assert rows[0]["matched"] is True
        assert rows[1]["matched"] is False
        assert rows[2]["matched"] is False
        assert len(result["unmatched_onset_times"]) == 1


# ── Summary statistics ────────────────────────────────────────────────────────

class TestSummary:

    def test_counts_all_matched(self):
        result = match_beats_to_onsets(_grid(1.0, 2.0), _onsets(1.0, 2.0), WIN)
        s = result["summary"]
        assert s["beats_total"]      == 2
        assert s["beats_matched"]    == 2
        assert s["beats_missed"]     == 0
        assert s["unmatched_onsets"] == 0

    def test_counts_partial_match(self):
        result = match_beats_to_onsets(_grid(1.0, 2.0, 3.0),
                                        _onsets(1.02, 5.0), WIN)
        s = result["summary"]
        assert s["beats_total"]      == 3
        assert s["beats_matched"]    == 1
        assert s["beats_missed"]     == 2
        assert s["unmatched_onsets"] == 1

    def test_mean_error_correct(self):
        # Errors: +20ms and -40ms → signed mean = -10ms; abs mean = 30ms
        result = match_beats_to_onsets(_grid(1.0, 2.0),
                                        _onsets(1.02, 1.96), WIN)
        s = result["summary"]
        assert s["mean_signed_error_ms"] == pytest.approx(-10.0)
        assert s["mean_abs_error_ms"]    == pytest.approx(30.0)

    def test_mean_error_none_when_no_matches(self):
        result = match_beats_to_onsets(_grid(1.0), _onsets(5.0), WIN)
        s = result["summary"]
        assert s["mean_signed_error_ms"] is None
        assert s["mean_abs_error_ms"]    is None

    def test_duplicate_onsets_summary_unmatched_count(self):
        """Two onsets near one beat → 1 matched, 1 unmatched onset."""
        result = match_beats_to_onsets(_grid(1.0), _onsets(1.01, 1.09), WIN)
        s = result["summary"]
        assert s["beats_matched"]    == 1
        assert s["unmatched_onsets"] == 1


# ── beat_grid helper ──────────────────────────────────────────────────────────

class TestBeatGrid:

    def test_length(self):
        assert len(beat_grid(60.0, 2, 16)) == 16

    def test_first_element_equals_count_in_time(self):
        bpm = 120.0
        count_in = 3
        beat_s = 60.0 / bpm
        g = beat_grid(bpm, count_in, 8)
        assert g[0] == pytest.approx(count_in * beat_s)

    def test_spacing_equals_beat_duration(self):
        bpm = 90.0
        beat_s = 60.0 / bpm
        g = beat_grid(bpm, 2, 8)
        diffs = np.diff(g)
        assert np.allclose(diffs, beat_s)

    def test_zero_count_in(self):
        bpm = 60.0
        g = beat_grid(bpm, 0, 4)
        assert g[0] == pytest.approx(0.0)
        assert g[1] == pytest.approx(1.0)
