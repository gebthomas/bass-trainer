"""Tests for summarize_evaluations() in scripts/plot_session_timeline.py.

All tests use hand-built TargetEvaluation objects — no audio hardware,
no file I/O, no matplotlib rendering.

Test matrix
-----------
Return type
  1.  Returns EvaluationSummary (NamedTuple)
  2.  Can be unpacked as a 9-tuple

Empty inputs
  3.  Empty evals + empty onsets → all zero counts, None errors
  4.  Empty evals + non-empty onsets → n_unmatched = len(onsets)

All on-time hits
  5.  n_on_time = n, n_miss = 0
  6.  No unmatched onsets when all matched
  7.  Mean signed error = 0 for exact hits
  8.  Mean abs error = 0 for exact hits
  9.  Max abs error = 0 for exact hits

Mixed classifications
  10. Counts per classification are correct
  11. Mean signed error excludes misses
  12. Mean abs error excludes misses
  13. Max abs error excludes misses

All misses
  14. n_miss = n_targets
  15. Error fields are None when all targets are misses

Unmatched (extra) onsets
  16. Counted correctly when some onsets are unmatched
  17. All unmatched when all evals are misses
  18. Zero unmatched when onset list is empty
  19. Counting is index-based, not position-based

Signed vs absolute error
  20. Mean signed is cancelled by opposite-sign errors
  21. Mean abs is not cancelled by direction
  22. Max abs picks the largest magnitude regardless of sign

Single-target edge cases
  23. Single exact hit
  24. Single late hit
  25. Single early hit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.realtime_evaluator import TargetEvaluation
from plot_session_timeline import EvaluationSummary, summarize_evaluations


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eval(
    idx: int,
    expected: float,
    actual: float | None,
    error_ms: float | None,
    cls: str,
    onset_idx: int | None,
) -> TargetEvaluation:
    return TargetEvaluation(
        target_index        = idx,
        expected_time_s     = expected,
        actual_time_s       = actual,
        signed_error_ms     = error_ms,
        classification      = cls,
        matched_onset_index = onset_idx,
    )


def _hit(idx: int, onset_idx: int, error_ms: float,
         on_time_threshold_ms: float = 30.0) -> TargetEvaluation:
    """Build a matched hit with explicit error and onset index."""
    if abs(error_ms) <= on_time_threshold_ms:
        cls = "on_time"
    elif error_ms > 0:
        cls = "late"
    else:
        cls = "early"
    expected = float(idx + 1)
    actual   = expected + error_ms / 1000.0
    return _eval(idx, expected, actual, error_ms, cls, onset_idx)


def _miss(idx: int) -> TargetEvaluation:
    return _eval(idx, float(idx + 1), None, None, "miss", None)


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_evaluation_summary(self):
        assert isinstance(summarize_evaluations([], []), EvaluationSummary)

    def test_unpacks_as_nine_tuple(self):
        s = summarize_evaluations([], [])
        (n_targets, n_on_time, n_early, n_late, n_miss,
         n_unmatched, mean_signed, mean_abs, max_abs) = s
        assert isinstance(n_targets, int)
        assert isinstance(n_on_time, int)


# ── Empty inputs ──────────────────────────────────────────────────────────────

class TestEmptyInputs:
    def test_empty_evals_all_zero_counts(self):
        s = summarize_evaluations([], [])
        assert s.n_targets          == 0
        assert s.n_on_time          == 0
        assert s.n_early            == 0
        assert s.n_late             == 0
        assert s.n_miss             == 0
        assert s.n_unmatched_onsets == 0

    def test_empty_evals_errors_are_none(self):
        s = summarize_evaluations([], [])
        assert s.mean_signed_error_ms is None
        assert s.mean_abs_error_ms    is None
        assert s.max_abs_error_ms     is None

    def test_empty_evals_with_onsets_all_unmatched(self):
        s = summarize_evaluations([], [1.0, 2.0, 3.0])
        assert s.n_unmatched_onsets == 3


# ── All on-time hits ──────────────────────────────────────────────────────────

class TestAllOnTimeHits:
    def test_counts(self):
        evals = [_hit(0, 0, 0.0), _hit(1, 1, 10.0), _hit(2, 2, -10.0)]
        s = summarize_evaluations(evals, [1.0, 2.0, 3.0])
        assert s.n_targets == 3
        assert s.n_on_time == 3
        assert s.n_early   == 0
        assert s.n_late    == 0
        assert s.n_miss    == 0

    def test_no_unmatched_when_all_matched(self):
        evals = [_hit(0, 0, 0.0), _hit(1, 1, 0.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.n_unmatched_onsets == 0

    def test_mean_signed_error_zero_for_exact_hits(self):
        evals = [_hit(0, 0, 0.0), _hit(1, 1, 0.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.mean_signed_error_ms == pytest.approx(0.0)

    def test_mean_abs_error_zero_for_exact_hits(self):
        evals = [_hit(0, 0, 0.0), _hit(1, 1, 0.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.mean_abs_error_ms == pytest.approx(0.0)

    def test_max_abs_error_zero_for_exact_hits(self):
        evals = [_hit(0, 0, 0.0), _hit(1, 1, 0.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.max_abs_error_ms == pytest.approx(0.0)


# ── Mixed classifications ─────────────────────────────────────────────────────

class TestMixedClassifications:
    # One on_time (10ms), one early (-50ms), one late (+50ms), one miss.
    def _evals(self) -> list[TargetEvaluation]:
        return [
            _hit(0, 0,  10.0),   # on_time
            _hit(1, 1, -50.0),   # early
            _hit(2, 2,  50.0),   # late
            _miss(3),
        ]

    def test_classification_counts(self):
        s = summarize_evaluations(self._evals(), [1.0, 2.0, 3.0])
        assert s.n_targets == 4
        assert s.n_on_time == 1
        assert s.n_early   == 1
        assert s.n_late    == 1
        assert s.n_miss    == 1

    def test_mean_signed_excludes_misses(self):
        # errors 10, -50, 50 → mean = 10/3
        s = summarize_evaluations(self._evals(), [1.0, 2.0, 3.0])
        assert s.mean_signed_error_ms == pytest.approx(10.0 / 3.0)

    def test_mean_abs_excludes_misses(self):
        # |errors| 10, 50, 50 → mean = 110/3
        s = summarize_evaluations(self._evals(), [1.0, 2.0, 3.0])
        assert s.mean_abs_error_ms == pytest.approx(110.0 / 3.0)

    def test_max_abs_excludes_misses(self):
        s = summarize_evaluations(self._evals(), [1.0, 2.0, 3.0])
        assert s.max_abs_error_ms == pytest.approx(50.0)


# ── All misses ────────────────────────────────────────────────────────────────

class TestAllMisses:
    def test_n_miss_equals_n_targets(self):
        evals = [_miss(0), _miss(1), _miss(2)]
        s = summarize_evaluations(evals, [])
        assert s.n_targets == 3
        assert s.n_miss    == 3
        assert s.n_on_time == 0

    def test_error_fields_are_none(self):
        evals = [_miss(0), _miss(1)]
        s = summarize_evaluations(evals, [])
        assert s.mean_signed_error_ms is None
        assert s.mean_abs_error_ms    is None
        assert s.max_abs_error_ms     is None


# ── Unmatched onsets ──────────────────────────────────────────────────────────

class TestUnmatchedOnsets:
    def test_extra_onsets_counted(self):
        # 2 matched onsets (indices 0, 1); 3rd onset (index 2) unmatched
        evals = [_hit(0, 0, 10.0), _hit(1, 1, -10.0)]
        s = summarize_evaluations(evals, [1.0, 2.0, 3.0])
        assert s.n_unmatched_onsets == 1

    def test_all_unmatched_when_all_misses(self):
        evals = [_miss(0), _miss(1)]
        s = summarize_evaluations(evals, [1.0, 2.0, 3.0])
        assert s.n_unmatched_onsets == 3

    def test_zero_unmatched_when_no_onset_list(self):
        evals = [_miss(0)]
        s = summarize_evaluations(evals, [])
        assert s.n_unmatched_onsets == 0

    def test_counting_uses_onset_index_not_list_position(self):
        # onset_index=2 is consumed; indices 0 and 1 remain free (3 onsets total)
        evals = [_hit(0, 2, 0.0)]
        s = summarize_evaluations(evals, [1.0, 2.0, 3.0])
        assert s.n_unmatched_onsets == 2


# ── Signed vs absolute error ──────────────────────────────────────────────────

class TestSignedVsAbsoluteError:
    def test_mean_signed_cancelled_by_opposite_errors(self):
        evals = [_hit(0, 0, 60.0), _hit(1, 1, -60.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.mean_signed_error_ms == pytest.approx(0.0)

    def test_mean_abs_not_cancelled(self):
        evals = [_hit(0, 0, 60.0), _hit(1, 1, -60.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.mean_abs_error_ms == pytest.approx(60.0)

    def test_max_abs_picks_largest_magnitude(self):
        evals = [_hit(0, 0, 10.0), _hit(1, 1, -70.0), _hit(2, 2, 40.0)]
        s = summarize_evaluations(evals, [1.0, 2.0, 3.0])
        assert s.max_abs_error_ms == pytest.approx(70.0)

    def test_max_abs_from_negative_error(self):
        evals = [_hit(0, 0, 10.0), _hit(1, 1, -75.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.max_abs_error_ms == pytest.approx(75.0)

    def test_positive_mean_signed_for_consistently_late(self):
        evals = [_hit(0, 0, 40.0), _hit(1, 1, 60.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.mean_signed_error_ms == pytest.approx(50.0)

    def test_negative_mean_signed_for_consistently_early(self):
        evals = [_hit(0, 0, -40.0), _hit(1, 1, -60.0)]
        s = summarize_evaluations(evals, [1.0, 2.0])
        assert s.mean_signed_error_ms == pytest.approx(-50.0)


# ── Single-target edge cases ──────────────────────────────────────────────────

class TestSingleTarget:
    def test_single_exact_hit(self):
        s = summarize_evaluations([_hit(0, 0, 0.0)], [1.0])
        assert s.n_targets            == 1
        assert s.n_on_time            == 1
        assert s.mean_signed_error_ms == pytest.approx(0.0)
        assert s.mean_abs_error_ms    == pytest.approx(0.0)
        assert s.max_abs_error_ms     == pytest.approx(0.0)

    def test_single_late_hit(self):
        s = summarize_evaluations([_hit(0, 0, 50.0)], [1.0])
        assert s.n_late               == 1
        assert s.mean_signed_error_ms == pytest.approx(50.0)
        assert s.mean_abs_error_ms    == pytest.approx(50.0)
        assert s.max_abs_error_ms     == pytest.approx(50.0)

    def test_single_early_hit(self):
        s = summarize_evaluations([_hit(0, 0, -40.0)], [1.0])
        assert s.n_early              == 1
        assert s.mean_signed_error_ms == pytest.approx(-40.0)
        assert s.mean_abs_error_ms    == pytest.approx(40.0)
        assert s.max_abs_error_ms     == pytest.approx(40.0)

    def test_single_miss(self):
        s = summarize_evaluations([_miss(0)], [])
        assert s.n_targets            == 1
        assert s.n_miss               == 1
        assert s.mean_signed_error_ms is None
        assert s.mean_abs_error_ms    is None
        assert s.max_abs_error_ms     is None
